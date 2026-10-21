from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Protocol, Sequence


class ContextItemType(StrEnum):
    TARGET = "target"
    PREVIOUS_CHUNK = "previous_chunk"
    NEXT_CHUNK = "next_chunk"
    TABLE = "table"
    CAPTION = "caption"
    SECTION_HEADER = "section_header"
    PAGE_CONTEXT = "page_context"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class ContextItem:
    context_id: str
    item_type: ContextItemType
    text: str
    payload: Any

    score: float = 0.0
    report_id: str | None = None
    page_number: int | None = None
    section_title: str | None = None

    source_candidate_id: str | None = None
    distance: int = 0

    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ExpandedContext:
    target_id: str
    target: ContextItem
    items: tuple[ContextItem, ...]
    combined_text: str
    report_id: str | None = None
    page_numbers: tuple[int, ...] = ()
    section_titles: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def context_count(self) -> int:
        return len(self.items)

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.combined_text) // 4)


@dataclass(slots=True, frozen=True)
class ContextExpanderConfig:
    previous_chunk_count: int = 1
    next_chunk_count: int = 1

    include_target: bool = True
    include_caption: bool = True
    include_table_context: bool = True
    include_section_header: bool = True
    include_same_page_context: bool = False

    maximum_context_items: int = 8
    maximum_total_characters: int = 12000
    maximum_item_characters: int = 4000

    preserve_report_boundary: bool = True
    preserve_section_boundary: bool = False

    minimum_source_score: float = 0.0

    def __post_init__(self) -> None:
        if self.previous_chunk_count < 0:
            raise ValueError(
                "previous_chunk_count negatif olamaz.",
            )

        if self.next_chunk_count < 0:
            raise ValueError(
                "next_chunk_count negatif olamaz.",
            )

        if self.maximum_context_items <= 0:
            raise ValueError(
                "maximum_context_items pozitif olmalıdır.",
            )

        if self.maximum_total_characters <= 0:
            raise ValueError(
                "maximum_total_characters pozitif olmalıdır.",
            )

        if self.maximum_item_characters <= 0:
            raise ValueError(
                "maximum_item_characters pozitif olmalıdır.",
            )

        if self.minimum_source_score < 0:
            raise ValueError(
                "minimum_source_score negatif olamaz.",
            )


class ChunkRepositoryProtocol(Protocol):
    def get_optional(self, chunk_id: str) -> Any | None:
        ...

    def get_context_window(
        self,
        chunk_id: str,
        *,
        previous_count: int = 1,
        next_count: int = 1,
        include_target: bool = True,
    ) -> Sequence[Any]:
        ...

    def find_by_page(
        self,
        report_id: str,
        page_number: int,
    ) -> Sequence[Any]:
        ...


class TableRepositoryProtocol(Protocol):
    def get_optional(self, table_id: str) -> Any | None:
        ...

    def find(
        self,
        *,
        report_id: str | None = None,
        page_number: int | None = None,
    ) -> Sequence[Any]:
        ...


class ContextExpansionError(RuntimeError):
    """Bağlam genişletme işlemlerinde oluşan temel hata."""


class ContextExpander:
    """
    Reranking sonrasında seçilen kanıtların çevresindeki bağlamı genişletir.

    Eklenebilecek bağlam türleri:
    - Önceki ve sonraki chunk'lar
    - Aynı sayfadaki ilgili içerikler
    - Tablo başlığı ve caption bilgileri
    - Tablo satırının ait olduğu tam tablo
    - Bölüm ve alt bölüm başlıkları

    Amaç, seçilmiş kanıtın anlam bütünlüğünü koruyarak cevap modeline
    yeterli fakat kontrollü büyüklükte bağlam sağlamaktır.
    """

    def __init__(
        self,
        chunk_repository: ChunkRepositoryProtocol | None = None,
        table_repository: TableRepositoryProtocol | None = None,
        *,
        config: ContextExpanderConfig | None = None,
    ) -> None:
        self.chunk_repository = chunk_repository
        self.table_repository = table_repository
        self.config = config or ContextExpanderConfig()

    def expand(
        self,
        candidate: Any,
        *,
        additional_items: Iterable[Any] = (),
    ) -> ExpandedContext:
        target = self._to_target_context_item(candidate)

        if target.score < self.config.minimum_source_score:
            raise ContextExpansionError(
                "Adayın skoru minimum_source_score eşiğinin altında.",
            )

        context_items: list[ContextItem] = []

        if self.config.include_target:
            context_items.append(target)

        if self._is_table_candidate(candidate, target.payload):
            context_items.extend(
                self._expand_table_context(
                    candidate,
                    target,
                ),
            )
        else:
            context_items.extend(
                self._expand_chunk_context(
                    candidate,
                    target,
                ),
            )

        if self.config.include_same_page_context:
            context_items.extend(
                self._expand_same_page_context(target),
            )

        for item in additional_items:
            try:
                context_items.append(
                    self._to_context_item(
                        item,
                        item_type=ContextItemType.UNKNOWN,
                        source_candidate_id=target.context_id,
                    ),
                )
            except ContextExpansionError:
                continue

        filtered_items = self._filter_context_items(
            target=target,
            items=context_items,
        )

        ordered_items = self._order_context_items(
            target=target,
            items=filtered_items,
        )

        limited_items = self._limit_context_items(
            ordered_items,
        )

        combined_text = self._build_combined_text(
            limited_items,
        )

        page_numbers = tuple(
            sorted(
                {
                    item.page_number
                    for item in limited_items
                    if item.page_number is not None
                },
            ),
        )

        section_titles = tuple(
            dict.fromkeys(
                item.section_title
                for item in limited_items
                if item.section_title
            ),
        )

        return ExpandedContext(
            target_id=target.context_id,
            target=target,
            items=tuple(limited_items),
            combined_text=combined_text,
            report_id=target.report_id,
            page_numbers=page_numbers,
            section_titles=section_titles,
            metadata={
                "item_count": len(limited_items),
                "character_count": len(combined_text),
                "token_estimate": max(
                    1,
                    len(combined_text) // 4,
                ),
                "contains_table_context": any(
                    item.item_type == ContextItemType.TABLE
                    for item in limited_items
                ),
                "contains_caption": any(
                    item.item_type == ContextItemType.CAPTION
                    for item in limited_items
                ),
                "contains_neighbor_chunks": any(
                    item.item_type
                    in {
                        ContextItemType.PREVIOUS_CHUNK,
                        ContextItemType.NEXT_CHUNK,
                    }
                    for item in limited_items
                ),
            },
        )

    def expand_many(
        self,
        candidates: Iterable[Any],
        *,
        maximum_candidates: int | None = None,
    ) -> list[ExpandedContext]:
        results: list[ExpandedContext] = []

        for candidate in candidates:
            if (
                maximum_candidates is not None
                and len(results) >= maximum_candidates
            ):
                break

            try:
                results.append(self.expand(candidate))
            except ContextExpansionError:
                continue

        return results

    def merge_expanded_contexts(
        self,
        contexts: Iterable[ExpandedContext],
        *,
        maximum_total_characters: int | None = None,
    ) -> str:
        character_limit = (
            maximum_total_characters
            or self.config.maximum_total_characters
        )

        if character_limit <= 0:
            raise ValueError(
                "maximum_total_characters pozitif olmalıdır.",
            )

        parts: list[str] = []
        seen_texts: set[str] = set()
        current_length = 0

        for context in contexts:
            cleaned_text = self._clean_text(
                context.combined_text,
            )

            normalized_text = self._normalize_text(
                cleaned_text,
            )

            if not normalized_text:
                continue

            if normalized_text in seen_texts:
                continue

            separator_length = 2 if parts else 0
            available_length = (
                character_limit
                - current_length
                - separator_length
            )

            if available_length <= 0:
                break

            if len(cleaned_text) > available_length:
                cleaned_text = self._truncate_text(
                    cleaned_text,
                    available_length,
                )

            if not cleaned_text:
                break

            seen_texts.add(normalized_text)
            parts.append(cleaned_text)

            current_length += (
                len(cleaned_text)
                + separator_length
            )

        return "\n\n".join(parts)

    def _expand_chunk_context(
        self,
        candidate: Any,
        target: ContextItem,
    ) -> list[ContextItem]:
        if self.chunk_repository is None:
            return self._expand_from_candidate_metadata(
                candidate,
                target,
            )

        chunk_id = self._extract_chunk_id(
            candidate,
            target.payload,
        )

        if chunk_id is None:
            return self._expand_from_candidate_metadata(
                candidate,
                target,
            )

        try:
            chunks = self.chunk_repository.get_context_window(
                chunk_id,
                previous_count=self.config.previous_chunk_count,
                next_count=self.config.next_chunk_count,
                include_target=False,
            )
        except Exception:
            return self._expand_from_candidate_metadata(
                candidate,
                target,
            )

        context_items: list[ContextItem] = []

        target_position = self._extract_position(
            target.payload,
        )

        for chunk in chunks:
            chunk_id_value = self._extract_identifier(chunk)

            if chunk_id_value == target.context_id:
                continue

            chunk_position = self._extract_position(chunk)
            distance = self._calculate_distance(
                target_position,
                chunk_position,
            )

            item_type = (
                ContextItemType.PREVIOUS_CHUNK
                if (
                    target_position is not None
                    and chunk_position is not None
                    and chunk_position < target_position
                )
                else ContextItemType.NEXT_CHUNK
            )

            context_items.append(
                self._to_context_item(
                    chunk,
                    item_type=item_type,
                    source_candidate_id=target.context_id,
                    distance=distance,
                ),
            )

        if self.config.include_section_header:
            section_header = self._build_section_header_item(
                target,
            )

            if section_header is not None:
                context_items.append(section_header)

        return context_items

    def _expand_table_context(
        self,
        candidate: Any,
        target: ContextItem,
    ) -> list[ContextItem]:
        context_items: list[ContextItem] = []

        table = self._extract_table_payload(
            candidate,
            target.payload,
        )

        if table is None:
            return context_items

        table_id = self._extract_identifier(table)

        if (
            table_id is not None
            and self.table_repository is not None
        ):
            try:
                stored_table = self.table_repository.get_optional(
                    table_id,
                )
            except Exception:
                stored_table = None

            if stored_table is not None:
                table = stored_table

        if self.config.include_table_context:
            table_text = self._build_table_text(table)

            if table_text:
                context_items.append(
                    ContextItem(
                        context_id=(
                            table_id
                            or f"{target.context_id}::table"
                        ),
                        item_type=ContextItemType.TABLE,
                        text=self._truncate_text(
                            table_text,
                            self.config.maximum_item_characters,
                        ),
                        payload=table,
                        score=target.score,
                        report_id=(
                            self._extract_report_id(table)
                            or target.report_id
                        ),
                        page_number=(
                            self._extract_page_number(table)
                            or target.page_number
                        ),
                        section_title=(
                            self._extract_section_title(table)
                            or target.section_title
                        ),
                        source_candidate_id=target.context_id,
                        distance=0,
                        metadata={
                            "result_level": "table",
                        },
                    ),
                )

        if self.config.include_caption:
            caption = self._extract_first_value(
                table,
                (
                    "caption",
                    "title",
                    "table_title",
                    "description",
                ),
            )

            cleaned_caption = self._clean_text(caption)

            if cleaned_caption:
                context_items.append(
                    ContextItem(
                        context_id=(
                            f"{table_id or target.context_id}::caption"
                        ),
                        item_type=ContextItemType.CAPTION,
                        text=cleaned_caption,
                        payload=table,
                        score=target.score,
                        report_id=target.report_id,
                        page_number=target.page_number,
                        section_title=target.section_title,
                        source_candidate_id=target.context_id,
                        distance=0,
                    ),
                )

        if self.config.include_section_header:
            section_header = self._build_section_header_item(
                target,
            )

            if section_header is not None:
                context_items.append(section_header)

        return context_items

    def _expand_same_page_context(
        self,
        target: ContextItem,
    ) -> list[ContextItem]:
        if (
            self.chunk_repository is None
            or target.report_id is None
            or target.page_number is None
        ):
            return []

        try:
            page_chunks = self.chunk_repository.find_by_page(
                target.report_id,
                target.page_number,
            )
        except Exception:
            return []

        items: list[ContextItem] = []

        for chunk in page_chunks:
            chunk_id = self._extract_identifier(chunk)

            if chunk_id == target.context_id:
                continue

            try:
                items.append(
                    self._to_context_item(
                        chunk,
                        item_type=ContextItemType.PAGE_CONTEXT,
                        source_candidate_id=target.context_id,
                    ),
                )
            except ContextExpansionError:
                continue

        return items

    def _expand_from_candidate_metadata(
        self,
        candidate: Any,
        target: ContextItem,
    ) -> list[ContextItem]:
        metadata = self._extract_metadata(candidate)
        context_items: list[ContextItem] = []

        previous_text = metadata.get("previous_chunk_text")

        if previous_text:
            context_items.append(
                ContextItem(
                    context_id=(
                        metadata.get("previous_chunk_id")
                        or f"{target.context_id}::previous"
                    ),
                    item_type=ContextItemType.PREVIOUS_CHUNK,
                    text=self._truncate_text(
                        self._clean_text(previous_text),
                        self.config.maximum_item_characters,
                    ),
                    payload=previous_text,
                    score=target.score,
                    report_id=target.report_id,
                    page_number=target.page_number,
                    section_title=target.section_title,
                    source_candidate_id=target.context_id,
                    distance=1,
                ),
            )

        next_text = metadata.get("next_chunk_text")

        if next_text:
            context_items.append(
                ContextItem(
                    context_id=(
                        metadata.get("next_chunk_id")
                        or f"{target.context_id}::next"
                    ),
                    item_type=ContextItemType.NEXT_CHUNK,
                    text=self._truncate_text(
                        self._clean_text(next_text),
                        self.config.maximum_item_characters,
                    ),
                    payload=next_text,
                    score=target.score,
                    report_id=target.report_id,
                    page_number=target.page_number,
                    section_title=target.section_title,
                    source_candidate_id=target.context_id,
                    distance=1,
                ),
            )

        if self.config.include_section_header:
            section_header = self._build_section_header_item(
                target,
            )

            if section_header is not None:
                context_items.append(section_header)

        return context_items

    def _build_section_header_item(
        self,
        target: ContextItem,
    ) -> ContextItem | None:
        if not target.section_title:
            return None

        return ContextItem(
            context_id=f"{target.context_id}::section",
            item_type=ContextItemType.SECTION_HEADER,
            text=target.section_title,
            payload=target.section_title,
            score=target.score,
            report_id=target.report_id,
            page_number=target.page_number,
            section_title=target.section_title,
            source_candidate_id=target.context_id,
            distance=0,
        )

    def _filter_context_items(
        self,
        *,
        target: ContextItem,
        items: Iterable[ContextItem],
    ) -> list[ContextItem]:
        filtered: list[ContextItem] = []
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()

        for item in items:
            if not item.text:
                continue

            if (
                self.config.preserve_report_boundary
                and target.report_id is not None
                and item.report_id is not None
                and item.report_id != target.report_id
            ):
                continue

            if (
                self.config.preserve_section_boundary
                and target.section_title
                and item.section_title
                and self._normalize_text(item.section_title)
                != self._normalize_text(target.section_title)
            ):
                continue

            normalized_text = self._normalize_text(item.text)

            if not normalized_text:
                continue

            if item.context_id in seen_ids:
                continue

            if normalized_text in seen_texts:
                continue

            seen_ids.add(item.context_id)
            seen_texts.add(normalized_text)
            filtered.append(item)

        return filtered

    def _order_context_items(
        self,
        *,
        target: ContextItem,
        items: Sequence[ContextItem],
    ) -> list[ContextItem]:
        type_order = {
            ContextItemType.SECTION_HEADER: 0,
            ContextItemType.PREVIOUS_CHUNK: 1,
            ContextItemType.CAPTION: 2,
            ContextItemType.TARGET: 3,
            ContextItemType.TABLE: 4,
            ContextItemType.NEXT_CHUNK: 5,
            ContextItemType.PAGE_CONTEXT: 6,
            ContextItemType.UNKNOWN: 99,
        }

        return sorted(
            items,
            key=lambda item: (
                type_order.get(item.item_type, 99),
                abs(item.distance),
                item.page_number
                if item.page_number is not None
                else 10**9,
                item.context_id,
            ),
        )

    def _limit_context_items(
        self,
        items: Sequence[ContextItem],
    ) -> list[ContextItem]:
        limited: list[ContextItem] = []
        total_characters = 0

        for item in items:
            if len(limited) >= self.config.maximum_context_items:
                break

            text = self._truncate_text(
                item.text,
                self.config.maximum_item_characters,
            )

            remaining = (
                self.config.maximum_total_characters
                - total_characters
            )

            if remaining <= 0:
                break

            if len(text) > remaining:
                text = self._truncate_text(
                    text,
                    remaining,
                )

            if not text:
                break

            limited.append(
                ContextItem(
                    context_id=item.context_id,
                    item_type=item.item_type,
                    text=text,
                    payload=item.payload,
                    score=item.score,
                    report_id=item.report_id,
                    page_number=item.page_number,
                    section_title=item.section_title,
                    source_candidate_id=item.source_candidate_id,
                    distance=item.distance,
                    metadata=item.metadata,
                ),
            )

            total_characters += len(text)

        return limited

    def _build_combined_text(
        self,
        items: Sequence[ContextItem],
    ) -> str:
        parts: list[str] = []

        for item in items:
            label = self._context_label(item)

            source_parts: list[str] = []

            if item.page_number is not None:
                source_parts.append(
                    f"sayfa={item.page_number}",
                )

            if item.section_title:
                source_parts.append(
                    f"bölüm={item.section_title}",
                )

            source_suffix = (
                f" ({', '.join(source_parts)})"
                if source_parts
                else ""
            )

            parts.append(
                f"[{label}{source_suffix}]\n{item.text}",
            )

        combined_text = "\n\n".join(parts)

        return self._truncate_text(
            combined_text,
            self.config.maximum_total_characters,
        )

    def _to_target_context_item(
        self,
        candidate: Any,
    ) -> ContextItem:
        candidate_id = self._extract_candidate_id(candidate)
        payload = self._extract_payload(candidate)

        text = self._extract_candidate_text(
            candidate,
            payload,
        )

        if not text:
            raise ContextExpansionError(
                f"'{candidate_id}' adayının bağlam metni boş.",
            )

        score = self._extract_score(candidate)

        report_id = (
            self._optional_string(
                self._extract_first_value(
                    candidate,
                    ("report_id",),
                ),
            )
            or self._extract_report_id(payload)
        )

        page_number = (
            self._optional_int(
                self._extract_first_value(
                    candidate,
                    (
                        "page_number",
                        "source_page",
                        "page",
                    ),
                ),
            )
            or self._extract_page_number(payload)
        )

        section_title = (
            self._optional_string(
                self._extract_first_value(
                    candidate,
                    (
                        "section_title",
                        "section",
                        "heading",
                    ),
                ),
            )
            or self._extract_section_title(payload)
        )

        return ContextItem(
            context_id=candidate_id,
            item_type=ContextItemType.TARGET,
            text=self._truncate_text(
                text,
                self.config.maximum_item_characters,
            ),
            payload=payload,
            score=score,
            report_id=report_id,
            page_number=page_number,
            section_title=section_title,
            source_candidate_id=candidate_id,
            distance=0,
            metadata=self._extract_metadata(candidate),
        )

    def _to_context_item(
        self,
        source: Any,
        *,
        item_type: ContextItemType,
        source_candidate_id: str,
        distance: int = 0,
    ) -> ContextItem:
        context_id = self._extract_identifier(source)

        if context_id is None:
            raise ContextExpansionError(
                "Bağlam öğesi için kimlik bulunamadı.",
            )

        text = self._extract_candidate_text(
            source,
            source,
        )

        if not text:
            raise ContextExpansionError(
                f"'{context_id}' bağlam öğesinin metni boş.",
            )

        return ContextItem(
            context_id=context_id,
            item_type=item_type,
            text=self._truncate_text(
                text,
                self.config.maximum_item_characters,
            ),
            payload=source,
            score=self._extract_score(source),
            report_id=self._extract_report_id(source),
            page_number=self._extract_page_number(source),
            section_title=self._extract_section_title(source),
            source_candidate_id=source_candidate_id,
            distance=distance,
            metadata=self._extract_metadata(source),
        )

    def _extract_candidate_id(
        self,
        source: Any,
    ) -> str:
        value = self._extract_identifier(source)

        if value is None:
            payload = self._extract_payload(source)
            value = self._extract_identifier(payload)

        if value is None:
            raise ContextExpansionError(
                "Bağlam genişletme adayında kimlik bulunamadı.",
            )

        return value

    def _extract_candidate_text(
        self,
        source: Any,
        payload: Any,
    ) -> str:
        value = self._extract_first_value(
            source,
            (
                "text",
                "matched_text",
                "content",
                "chunk_text",
                "combined_text",
            ),
        )

        if value is not None:
            return self._clean_text(value)

        value = self._extract_first_value(
            payload,
            (
                "text",
                "matched_text",
                "content",
                "chunk_text",
            ),
        )

        if value is not None:
            return self._clean_text(value)

        table_text = self._build_table_text(payload)

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
            value = ContextExpander._extract_first_value(
                source,
                (name,),
            )

            if value is not None:
                return value

        return source

    @staticmethod
    def _extract_score(source: Any) -> float:
        value = ContextExpander._extract_first_value(
            source,
            (
                "score",
                "rerank_score",
                "retrieval_score",
                "confidence",
            ),
        )

        if value is None:
            return 0.0

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _extract_identifier(
        cls,
        source: Any,
    ) -> str | None:
        value = cls._extract_first_value(
            source,
            (
                "candidate_id",
                "document_id",
                "chunk_id",
                "table_id",
                "context_id",
                "id",
                "identifier",
            ),
        )

        if value is None:
            metadata = cls._extract_metadata(source)

            for key in (
                "candidate_id",
                "document_id",
                "chunk_id",
                "table_id",
                "id",
            ):
                if metadata.get(key) is not None:
                    value = metadata[key]
                    break

        if value is None:
            return None

        cleaned = str(value).strip()
        return cleaned or None

    @classmethod
    def _extract_chunk_id(
        cls,
        candidate: Any,
        payload: Any,
    ) -> str | None:
        for source in (candidate, payload):
            value = cls._extract_first_value(
                source,
                (
                    "chunk_id",
                    "document_id",
                    "candidate_id",
                    "id",
                ),
            )

            if value is not None:
                cleaned = str(value).strip()

                if "::row:" not in cleaned:
                    return cleaned

        return None

    @classmethod
    def _extract_table_payload(
        cls,
        candidate: Any,
        payload: Any,
    ) -> Any | None:
        table = cls._extract_first_value(
            candidate,
            ("table",),
        )

        if table is not None:
            return table

        if cls._is_table_candidate(candidate, payload):
            return payload

        return None

    @classmethod
    def _is_table_candidate(
        cls,
        candidate: Any,
        payload: Any,
    ) -> bool:
        if cls._extract_first_value(
            candidate,
            ("row_index", "table_id"),
        ) is not None:
            return True

        if cls._extract_first_value(
            payload,
            (
                "rows",
                "cells",
                "headers",
                "column_headers",
                "table_id",
            ),
        ) is not None:
            return True

        metadata = cls._extract_metadata(candidate)

        return metadata.get("result_level") in {
            "table",
            "row",
        }

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
    def _extract_position(
        cls,
        source: Any,
    ) -> int | None:
        value = cls._extract_first_value(
            source,
            (
                "chunk_position",
                "chunk_index",
                "position",
                "sequence_number",
                "order",
            ),
        )

        if value is None:
            metadata = cls._extract_metadata(source)

            for key in (
                "chunk_position",
                "chunk_index",
                "position",
                "sequence_number",
                "order",
            ):
                if metadata.get(key) is not None:
                    value = metadata[key]
                    break

        return cls._optional_int(value)

    @staticmethod
    def _calculate_distance(
        target_position: int | None,
        candidate_position: int | None,
    ) -> int:
        if (
            target_position is None
            or candidate_position is None
        ):
            return 1

        return abs(
            candidate_position - target_position,
        )

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
    def _extract_metadata(source: Any) -> dict[str, Any]:
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

            if value is not None:
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
    def _context_label(
        item: ContextItem,
    ) -> str:
        labels = {
            ContextItemType.TARGET: "ANA KANIT",
            ContextItemType.PREVIOUS_CHUNK: "ÖNCEKİ BAĞLAM",
            ContextItemType.NEXT_CHUNK: "SONRAKİ BAĞLAM",
            ContextItemType.TABLE: "TABLO BAĞLAMI",
            ContextItemType.CAPTION: "TABLO/ŞEKİL BAŞLIĞI",
            ContextItemType.SECTION_HEADER: "BÖLÜM BAŞLIĞI",
            ContextItemType.PAGE_CONTEXT: "AYNI SAYFA BAĞLAMI",
            ContextItemType.UNKNOWN: "EK BAĞLAM",
        }

        return labels[item.item_type]

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()
        return cleaned or None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _truncate_text(
        text: str,
        maximum_length: int,
    ) -> str:
        cleaned = ContextExpander._clean_text(text)

        if maximum_length <= 0:
            return ""

        if len(cleaned) <= maximum_length:
            return cleaned

        if maximum_length <= 3:
            return cleaned[:maximum_length]

        return (
            cleaned[: maximum_length - 3].rstrip()
            + "..."
        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""

        enum_value = getattr(value, "value", value)

        return " ".join(
            str(enum_value).replace("\n", " ").split(),
        )

    @staticmethod
    def _normalize_text(value: str) -> str:
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

        return " ".join(normalized.split())


__all__ = [
    "ChunkRepositoryProtocol",
    "ContextExpander",
    "ContextExpanderConfig",
    "ContextExpansionError",
    "ContextItem",
    "ContextItemType",
    "ExpandedContext",
    "TableRepositoryProtocol",
]