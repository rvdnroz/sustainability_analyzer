from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from models.chunk import Chunk


class ChunkRepositoryError(RuntimeError):
    """Chunk repository işlemleri sırasında oluşan temel hata."""


class ChunkNotFoundError(ChunkRepositoryError):
    """İstenen chunk repository içerisinde bulunamadığında yükseltilir."""


class DuplicateChunkError(ChunkRepositoryError):
    """Aynı kimliğe sahip chunk yeniden eklenmek istendiğinde yükseltilir."""


@dataclass(slots=True, frozen=True)
class ChunkNeighbor:
    chunk_id: str
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None


@dataclass(slots=True, frozen=True)
class ChunkSearchFilter:
    report_id: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    category: str | None = None
    standards: Sequence[str] = ()
    mentioned_years: Sequence[int] = ()


class ChunkRepository:
    """
    Chunk kayıtlarını ve chunk komşuluk ilişkilerini bellekte saklar.

    Repository şu işlemleri destekler:
    - Tekli ve toplu chunk ekleme
    - Chunk kimliğiyle erişim
    - Rapor, sayfa, bölüm ve kategori bazlı filtreleme
    - Önceki ve sonraki chunk ilişkilerini saklama
    - Komşu chunk'ları belirli bir pencereyle getirme
    - JSON dosyasına kalıcılaştırma ve geri yükleme

    Kalıcı veri tabanı katmanı oluşturulduğunda bu sınıfın arayüzü korunarak
    SQLite veya başka bir veri tabanı implementasyonuna geçirilebilir.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._report_index: dict[str, set[str]] = defaultdict(set)
        self._page_index: dict[tuple[str, int], set[str]] = defaultdict(set)
        self._section_index: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._category_index: dict[str, set[str]] = defaultdict(set)
        self._neighbors: dict[str, ChunkNeighbor] = {}
        self._lock = RLock()

    def add(
        self,
        chunk: Chunk,
        *,
        overwrite: bool = False,
    ) -> Chunk:
        chunk_id = self._get_chunk_id(chunk)

        with self._lock:
            if chunk_id in self._chunks and not overwrite:
                raise DuplicateChunkError(
                    f"'{chunk_id}' kimliğine sahip chunk zaten kayıtlı.",
                )

            if chunk_id in self._chunks:
                self._remove_from_indexes(self._chunks[chunk_id])

            stored_chunk = deepcopy(chunk)
            self._chunks[chunk_id] = stored_chunk
            self._add_to_indexes(stored_chunk)
            self._register_chunk_neighbors(stored_chunk)

            return deepcopy(stored_chunk)

    def add_many(
        self,
        chunks: Iterable[Chunk],
        *,
        overwrite: bool = False,
        infer_neighbors: bool = True,
    ) -> list[Chunk]:
        chunk_list = list(chunks)
        if not chunk_list:
            return []

        if infer_neighbors:
            chunk_list = self._infer_sequential_neighbors(chunk_list)

        added_chunks: list[Chunk] = []

        with self._lock:
            for chunk in chunk_list:
                added_chunks.append(
                    self.add(
                        chunk,
                        overwrite=overwrite,
                    ),
                )

        return added_chunks

    def get(self, chunk_id: str) -> Chunk:
        normalized_id = self._normalize_identifier(chunk_id)

        with self._lock:
            chunk = self._chunks.get(normalized_id)

            if chunk is None:
                raise ChunkNotFoundError(
                    f"'{normalized_id}' kimliğine sahip chunk bulunamadı.",
                )

            return deepcopy(chunk)

    def get_optional(self, chunk_id: str) -> Chunk | None:
        normalized_id = self._normalize_identifier(chunk_id)

        with self._lock:
            chunk = self._chunks.get(normalized_id)
            return deepcopy(chunk) if chunk is not None else None

    def exists(self, chunk_id: str) -> bool:
        normalized_id = self._normalize_identifier(chunk_id)

        with self._lock:
            return normalized_id in self._chunks

    def update(self, chunk: Chunk) -> Chunk:
        chunk_id = self._get_chunk_id(chunk)

        with self._lock:
            if chunk_id not in self._chunks:
                raise ChunkNotFoundError(
                    f"'{chunk_id}' kimliğine sahip chunk güncellenemedi; "
                    "kayıt bulunamadı.",
                )

            return self.add(chunk, overwrite=True)

    def delete(self, chunk_id: str) -> Chunk:
        normalized_id = self._normalize_identifier(chunk_id)

        with self._lock:
            chunk = self._chunks.pop(normalized_id, None)

            if chunk is None:
                raise ChunkNotFoundError(
                    f"'{normalized_id}' kimliğine sahip chunk bulunamadı.",
                )

            self._remove_from_indexes(chunk)
            self._unlink_neighbors(normalized_id)

            return deepcopy(chunk)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._report_index.clear()
            self._page_index.clear()
            self._section_index.clear()
            self._category_index.clear()
            self._neighbors.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def all(self) -> list[Chunk]:
        with self._lock:
            return [
                deepcopy(chunk)
                for chunk in self._sort_chunks(self._chunks.values())
            ]

    def iter_chunks(self) -> Iterator[Chunk]:
        for chunk in self.all():
            yield chunk

    def find(
        self,
        *,
        filters: ChunkSearchFilter | None = None,
        text_query: str | None = None,
        limit: int | None = None,
    ) -> list[Chunk]:
        filters = filters or ChunkSearchFilter()

        with self._lock:
            candidate_ids = self._resolve_candidate_ids(filters)
            candidates = [
                self._chunks[chunk_id]
                for chunk_id in candidate_ids
                if chunk_id in self._chunks
            ]

            candidates = [
                chunk
                for chunk in candidates
                if self._matches_filter(chunk, filters)
            ]

            if text_query is not None and text_query.strip():
                candidates = self._filter_by_text(
                    candidates,
                    text_query,
                )

            sorted_chunks = self._sort_chunks(candidates)

            if limit is not None:
                if limit < 0:
                    raise ValueError("limit negatif olamaz.")

                sorted_chunks = sorted_chunks[:limit]

            return [deepcopy(chunk) for chunk in sorted_chunks]

    def find_by_report(self, report_id: str) -> list[Chunk]:
        return self.find(
            filters=ChunkSearchFilter(report_id=report_id),
        )

    def find_by_page(
        self,
        report_id: str,
        page_number: int,
    ) -> list[Chunk]:
        return self.find(
            filters=ChunkSearchFilter(
                report_id=report_id,
                page_number=page_number,
            ),
        )

    def find_by_section(
        self,
        report_id: str,
        section_title: str,
    ) -> list[Chunk]:
        return self.find(
            filters=ChunkSearchFilter(
                report_id=report_id,
                section_title=section_title,
            ),
        )

    def find_by_category(self, category: str) -> list[Chunk]:
        return self.find(
            filters=ChunkSearchFilter(category=category),
        )

    def set_neighbors(
        self,
        chunk_id: str,
        *,
        previous_chunk_id: str | None = None,
        next_chunk_id: str | None = None,
        validate_references: bool = True,
    ) -> ChunkNeighbor:
        normalized_chunk_id = self._normalize_identifier(chunk_id)
        normalized_previous_id = self._normalize_optional_identifier(
            previous_chunk_id,
        )
        normalized_next_id = self._normalize_optional_identifier(
            next_chunk_id,
        )

        with self._lock:
            if normalized_chunk_id not in self._chunks:
                raise ChunkNotFoundError(
                    f"'{normalized_chunk_id}' kimliğine sahip chunk bulunamadı.",
                )

            if validate_references:
                for neighbor_id in (
                    normalized_previous_id,
                    normalized_next_id,
                ):
                    if (
                        neighbor_id is not None
                        and neighbor_id not in self._chunks
                    ):
                        raise ChunkNotFoundError(
                            f"Komşu chunk '{neighbor_id}' bulunamadı.",
                        )

            if normalized_chunk_id in {
                normalized_previous_id,
                normalized_next_id,
            }:
                raise ChunkRepositoryError(
                    "Bir chunk kendisinin komşusu olamaz.",
                )

            neighbor = ChunkNeighbor(
                chunk_id=normalized_chunk_id,
                previous_chunk_id=normalized_previous_id,
                next_chunk_id=normalized_next_id,
            )
            self._neighbors[normalized_chunk_id] = neighbor

            self._update_chunk_neighbor_metadata(neighbor)
            self._create_reverse_neighbor_links(neighbor)

            return neighbor

    def get_neighbor_info(self, chunk_id: str) -> ChunkNeighbor:
        normalized_id = self._normalize_identifier(chunk_id)

        with self._lock:
            if normalized_id not in self._chunks:
                raise ChunkNotFoundError(
                    f"'{normalized_id}' kimliğine sahip chunk bulunamadı.",
                )

            return self._neighbors.get(
                normalized_id,
                ChunkNeighbor(chunk_id=normalized_id),
            )

    def get_previous(self, chunk_id: str) -> Chunk | None:
        neighbor = self.get_neighbor_info(chunk_id)

        if neighbor.previous_chunk_id is None:
            return None

        return self.get_optional(neighbor.previous_chunk_id)

    def get_next(self, chunk_id: str) -> Chunk | None:
        neighbor = self.get_neighbor_info(chunk_id)

        if neighbor.next_chunk_id is None:
            return None

        return self.get_optional(neighbor.next_chunk_id)

    def get_context_window(
        self,
        chunk_id: str,
        *,
        previous_count: int = 1,
        next_count: int = 1,
        include_target: bool = True,
    ) -> list[Chunk]:
        if previous_count < 0 or next_count < 0:
            raise ValueError(
                "previous_count ve next_count negatif olamaz.",
            )

        normalized_id = self._normalize_identifier(chunk_id)

        with self._lock:
            if normalized_id not in self._chunks:
                raise ChunkNotFoundError(
                    f"'{normalized_id}' kimliğine sahip chunk bulunamadı.",
                )

            previous_chunks: list[Chunk] = []
            current_id = normalized_id

            for _ in range(previous_count):
                neighbor = self._neighbors.get(current_id)
                if neighbor is None or neighbor.previous_chunk_id is None:
                    break

                previous_id = neighbor.previous_chunk_id
                previous_chunk = self._chunks.get(previous_id)

                if previous_chunk is None:
                    break

                previous_chunks.append(previous_chunk)
                current_id = previous_id

            previous_chunks.reverse()

            next_chunks: list[Chunk] = []
            current_id = normalized_id

            for _ in range(next_count):
                neighbor = self._neighbors.get(current_id)
                if neighbor is None or neighbor.next_chunk_id is None:
                    break

                next_id = neighbor.next_chunk_id
                next_chunk = self._chunks.get(next_id)

                if next_chunk is None:
                    break

                next_chunks.append(next_chunk)
                current_id = next_id

            result: list[Chunk] = previous_chunks

            if include_target:
                result.append(self._chunks[normalized_id])

            result.extend(next_chunks)

            return [deepcopy(chunk) for chunk in result]

    def load_from_database(
        self,
        database: Any,
        *,
        report_database_id: int,
        report_uid: str,
        clear_existing: bool = True,
    ) -> int:
        """
        SQLite chunks kayıtlarını yeniden Chunk nesnelerine dönüştürür.

        `--skip-ingestion` yeni bir Python süreci başlattığı için bellek içi
        repository boştur. Bu metot kalıcı chunks tablosunu yeniden yükler.
        """

        if report_database_id < 1:
            raise ValueError("report_database_id pozitif olmalıdır.")

        normalized_report_uid = str(report_uid or "").strip()
        if not normalized_report_uid:
            raise ValueError("report_uid boş olamaz.")

        records = list(
            database.get_chunks(report_database_id)
            or []
        )

        chunks: list[Chunk] = []

        for record in records:
            try:
                chunks.append(
                    self._chunk_from_database_record(
                        record,
                        report_uid=normalized_report_uid,
                    )
                )
            except Exception as exc:
                raise ChunkRepositoryError(
                    "SQLite chunk kaydı modele dönüştürülemedi: "
                    f"chunk_uid={record.get('chunk_uid')!r}, hata={exc}"
                ) from exc

        with self._lock:
            if clear_existing:
                self.clear()

            self.add_many(
                chunks,
                overwrite=not clear_existing,
                infer_neighbors=False,
            )

            for record in records:
                chunk_uid = str(
                    record.get("chunk_uid")
                    or ""
                ).strip()
                if not chunk_uid or not self.exists(chunk_uid):
                    continue

                self.set_neighbors(
                    chunk_uid,
                    previous_chunk_id=record.get(
                        "previous_chunk_uid"
                    ),
                    next_chunk_id=record.get(
                        "next_chunk_uid"
                    ),
                    validate_references=False,
                )

        return len(chunks)

    @classmethod
    def _chunk_from_database_record(
        cls,
        record: Mapping[str, Any],
        *,
        report_uid: str,
    ) -> Chunk:
        metadata = dict(record.get("metadata") or {})

        chunk_uid = str(
            record.get("chunk_uid")
            or metadata.get("chunk_id")
            or ""
        ).strip()

        page_number = cls._to_int(
            record.get("page_number")
        ) or 1
        chunk_index = cls._to_int(
            record.get("chunk_index")
        ) or 0
        text_content = str(
            record.get("text_content")
            or ""
        )

        metadata.update(
            {
                "chunk_id": chunk_uid,
                "report_id": report_uid,
                "page_number": page_number,
                "page_start": page_number,
                "page_end": page_number,
                "chunk_index": chunk_index,
                "chunk_position": chunk_index,
                "section_title": record.get(
                    "section_title"
                ),
                "subsection_title": record.get(
                    "subsection_title"
                ),
                "category": record.get("category"),
                "language": record.get("language"),
                "previous_chunk_id": record.get(
                    "previous_chunk_uid"
                ),
                "next_chunk_id": record.get(
                    "next_chunk_uid"
                ),
                "content_hash": record.get(
                    "content_hash"
                ),
            }
        )

        chunk_uuid = uuid5(
            NAMESPACE_URL,
            f"sustainability-chunk:{chunk_uid}",
        )
        report_uuid = uuid5(
            NAMESPACE_URL,
            f"sustainability-report:{report_uid}",
        )

        raw_source_type = str(
            metadata.get("source_type")
            or ""
        ).strip().casefold()

        source_type_aliases = {
            "native": "native_text",
            "native_text": "native_text",
            "text": "native_text",
            "ocr": "ocr_text",
            "ocr_text": "ocr_text",
            "visual_ocr": "ocr_text",
            "table": "table_context",
            "table_context": "table_context",
            "standard": "standard_index",
            "standard_index": "standard_index",
            "mixed": "mixed",
            "unknown": "unknown",
        }
        normalized_source_type = source_type_aliases.get(
            raw_source_type,
            "unknown",
        )

        # Model UUID kullanırken repository ve filtreleme katmanı eski,
        # okunabilir kimliklerle çalışmaya devam eder.
        metadata.update(
            {
                "chunk_id": chunk_uid,
                "original_chunk_uid": chunk_uid,
                "report_id": report_uid,
                "original_report_uid": report_uid,
                "model_chunk_uuid": str(chunk_uuid),
                "model_report_uuid": str(report_uuid),
                "source_type": normalized_source_type,
            }
        )

        payload = {
            "chunk_id": chunk_uid,
            "id": chunk_uuid,
            "identifier": chunk_uid,
            "report_id": report_uuid,
            "chunk_index": chunk_index,
            "start_page": page_number,
            "end_page": page_number,
            "page_start": page_number,
            "page_end": page_number,
            "page_number": page_number,
            "text": text_content,
            "content": text_content,
            "chunk_text": text_content,
            "normalized_text": text_content,
            "searchable_text": text_content,
            "contextualized_text": text_content,
            "section_title": record.get(
                "section_title"
            ) or "",
            "subsection_title": record.get(
                "subsection_title"
            ),
            "category": record.get("category"),
            "language": record.get("language"),
            "source_type": normalized_source_type,
            "content_hash": record.get(
                "content_hash"
            ) or "",
            "metadata": metadata,
        }

        model_fields = getattr(Chunk, "model_fields", None)
        if isinstance(model_fields, Mapping):
            payload = {
                key: value
                for key, value in payload.items()
                if key in model_fields
            }
        elif is_dataclass(Chunk):
            allowed = {
                item.name
                for item in fields(Chunk)
            }
            payload = {
                key: value
                for key, value in payload.items()
                if key in allowed
            }

        return cls._chunk_from_dict(payload)

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            payload = {
                "version": 1,
                "chunks": [
                    self._chunk_to_dict(chunk)
                    for chunk in self._sort_chunks(self._chunks.values())
                ],
                "neighbors": [
                    {
                        "chunk_id": neighbor.chunk_id,
                        "previous_chunk_id": neighbor.previous_chunk_id,
                        "next_chunk_id": neighbor.next_chunk_id,
                    }
                    for neighbor in self._neighbors.values()
                ],
            }

        temporary_path = output_path.with_suffix(
            f"{output_path.suffix}.tmp",
        )

        try:
            temporary_path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    default=self._json_default,
                ),
                encoding="utf-8",
            )
            temporary_path.replace(output_path)
        except (OSError, TypeError, ValueError) as exc:
            temporary_path.unlink(missing_ok=True)
            raise ChunkRepositoryError(
                f"Chunk repository JSON dosyasına yazılamadı: {output_path}",
            ) from exc

        return output_path

    def load_json(
        self,
        path: str | Path,
        *,
        clear_existing: bool = True,
    ) -> int:
        input_path = Path(path).expanduser().resolve()

        if not input_path.is_file():
            raise FileNotFoundError(
                f"Chunk repository dosyası bulunamadı: {input_path}",
            )

        try:
            payload = json.loads(
                input_path.read_text(encoding="utf-8"),
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ChunkRepositoryError(
                f"Chunk repository dosyası okunamadı: {input_path}",
            ) from exc

        chunk_records = payload.get("chunks", [])
        neighbor_records = payload.get("neighbors", [])

        if not isinstance(chunk_records, list):
            raise ChunkRepositoryError(
                "JSON içerisindeki 'chunks' alanı liste olmalıdır.",
            )

        chunks = [
            self._chunk_from_dict(record)
            for record in chunk_records
        ]

        with self._lock:
            if clear_existing:
                self.clear()

            self.add_many(
                chunks,
                overwrite=not clear_existing,
                infer_neighbors=False,
            )

            for record in neighbor_records:
                if not isinstance(record, Mapping):
                    continue

                chunk_id = record.get("chunk_id")
                if not chunk_id or not self.exists(str(chunk_id)):
                    continue

                self.set_neighbors(
                    str(chunk_id),
                    previous_chunk_id=record.get(
                        "previous_chunk_id",
                    ),
                    next_chunk_id=record.get("next_chunk_id"),
                    validate_references=False,
                )

        return len(chunks)

    def _resolve_candidate_ids(
        self,
        filters: ChunkSearchFilter,
    ) -> set[str]:
        index_sets: list[set[str]] = []

        if filters.report_id is not None:
            report_id = str(filters.report_id).strip()
            index_sets.append(
                set(self._report_index.get(report_id, set())),
            )

        if (
            filters.report_id is not None
            and filters.page_number is not None
        ):
            page_key = (
                str(filters.report_id).strip(),
                int(filters.page_number),
            )
            index_sets.append(
                set(self._page_index.get(page_key, set())),
            )

        if (
            filters.report_id is not None
            and filters.section_title is not None
        ):
            section_key = (
                str(filters.report_id).strip(),
                self._normalize_text(filters.section_title),
            )
            index_sets.append(
                set(self._section_index.get(section_key, set())),
            )

        if filters.category is not None:
            category = self._normalize_text(filters.category)
            index_sets.append(
                set(self._category_index.get(category, set())),
            )

        if not index_sets:
            return set(self._chunks)

        candidate_ids = index_sets[0]

        for index_set in index_sets[1:]:
            candidate_ids.intersection_update(index_set)

        return candidate_ids

    def _matches_filter(
        self,
        chunk: Chunk,
        filters: ChunkSearchFilter,
    ) -> bool:
        metadata = self._get_metadata(chunk)

        if filters.report_id is not None:
            report_id = self._get_value(
                chunk,
                metadata,
                ("report_id",),
            )
            if str(report_id or "").strip() != str(
                filters.report_id,
            ).strip():
                return False

        if filters.page_number is not None:
            page_number = self._get_value(
                chunk,
                metadata,
                ("page_number", "source_page", "page"),
            )
            if self._to_int(page_number) != filters.page_number:
                return False

        if filters.section_title is not None:
            section_title = self._get_value(
                chunk,
                metadata,
                ("section_title", "section", "heading"),
            )
            if self._normalize_text(section_title) != self._normalize_text(
                filters.section_title,
            ):
                return False

        if filters.category is not None:
            category = self._get_value(
                chunk,
                metadata,
                ("category",),
            )
            if self._normalize_text(category) != self._normalize_text(
                filters.category,
            ):
                return False

        if filters.standards:
            chunk_standards = self._as_normalized_set(
                self._get_value(
                    chunk,
                    metadata,
                    (
                        "standards",
                        "standard_codes",
                        "standard_references",
                    ),
                ),
            )
            required_standards = self._as_normalized_set(
                filters.standards,
            )

            if not required_standards.issubset(chunk_standards):
                return False

        if filters.mentioned_years:
            chunk_years = {
                year
                for year in (
                    self._to_int(value)
                    for value in self._as_sequence(
                        metadata.get("mentioned_years"),
                    )
                )
                if year is not None
            }
            required_years = {
                int(year)
                for year in filters.mentioned_years
            }

            if not required_years.issubset(chunk_years):
                return False

        return True

    def _add_to_indexes(self, chunk: Chunk) -> None:
        chunk_id = self._get_chunk_id(chunk)
        metadata = self._get_metadata(chunk)

        report_id = self._get_value(
            chunk,
            metadata,
            ("report_id",),
        )
        page_number = self._get_value(
            chunk,
            metadata,
            ("page_number", "source_page", "page"),
        )
        section_title = self._get_value(
            chunk,
            metadata,
            ("section_title", "section", "heading"),
        )
        category = self._get_value(
            chunk,
            metadata,
            ("category",),
        )

        if report_id is not None:
            normalized_report_id = str(report_id).strip()
            self._report_index[normalized_report_id].add(chunk_id)

            normalized_page_number = self._to_int(page_number)
            if normalized_page_number is not None:
                self._page_index[
                    (normalized_report_id, normalized_page_number)
                ].add(chunk_id)

            if section_title is not None:
                self._section_index[
                    (
                        normalized_report_id,
                        self._normalize_text(section_title),
                    )
                ].add(chunk_id)

        if category is not None:
            self._category_index[
                self._normalize_text(category)
            ].add(chunk_id)

    def _remove_from_indexes(self, chunk: Chunk) -> None:
        chunk_id = self._get_chunk_id(chunk)

        for index in (
            self._report_index,
            self._page_index,
            self._section_index,
            self._category_index,
        ):
            empty_keys: list[Any] = []

            for key, chunk_ids in index.items():
                chunk_ids.discard(chunk_id)

                if not chunk_ids:
                    empty_keys.append(key)

            for key in empty_keys:
                del index[key]

    def _register_chunk_neighbors(self, chunk: Chunk) -> None:
        chunk_id = self._get_chunk_id(chunk)
        metadata = self._get_metadata(chunk)

        previous_chunk_id = self._get_value(
            chunk,
            metadata,
            ("previous_chunk_id", "prev_chunk_id"),
        )
        next_chunk_id = self._get_value(
            chunk,
            metadata,
            ("next_chunk_id",),
        )

        self._neighbors[chunk_id] = ChunkNeighbor(
            chunk_id=chunk_id,
            previous_chunk_id=self._normalize_optional_identifier(
                previous_chunk_id,
            ),
            next_chunk_id=self._normalize_optional_identifier(
                next_chunk_id,
            ),
        )

    def _infer_sequential_neighbors(
        self,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        sorted_chunks = self._sort_chunks(chunks)
        enriched_chunks: list[Chunk] = []

        for index, chunk in enumerate(sorted_chunks):
            previous_chunk_id = (
                self._get_chunk_id(sorted_chunks[index - 1])
                if index > 0
                else None
            )
            next_chunk_id = (
                self._get_chunk_id(sorted_chunks[index + 1])
                if index < len(sorted_chunks) - 1
                else None
            )

            enriched_chunks.append(
                self._set_chunk_metadata_values(
                    chunk,
                    {
                        "previous_chunk_id": previous_chunk_id,
                        "next_chunk_id": next_chunk_id,
                        "chunk_position": index,
                        "chunk_number": index + 1,
                        "total_chunks": len(sorted_chunks),
                    },
                ),
            )

        return enriched_chunks

    def _unlink_neighbors(self, chunk_id: str) -> None:
        removed_neighbor = self._neighbors.pop(chunk_id, None)

        if removed_neighbor is None:
            return

        if removed_neighbor.previous_chunk_id is not None:
            previous_neighbor = self._neighbors.get(
                removed_neighbor.previous_chunk_id,
            )

            if previous_neighbor is not None:
                updated_previous = ChunkNeighbor(
                    chunk_id=previous_neighbor.chunk_id,
                    previous_chunk_id=previous_neighbor.previous_chunk_id,
                    next_chunk_id=removed_neighbor.next_chunk_id,
                )
                self._neighbors[updated_previous.chunk_id] = updated_previous
                self._update_chunk_neighbor_metadata(updated_previous)

        if removed_neighbor.next_chunk_id is not None:
            next_neighbor = self._neighbors.get(
                removed_neighbor.next_chunk_id,
            )

            if next_neighbor is not None:
                updated_next = ChunkNeighbor(
                    chunk_id=next_neighbor.chunk_id,
                    previous_chunk_id=removed_neighbor.previous_chunk_id,
                    next_chunk_id=next_neighbor.next_chunk_id,
                )
                self._neighbors[updated_next.chunk_id] = updated_next
                self._update_chunk_neighbor_metadata(updated_next)

    def _create_reverse_neighbor_links(
        self,
        neighbor: ChunkNeighbor,
    ) -> None:
        if neighbor.previous_chunk_id is not None:
            previous = self._neighbors.get(
                neighbor.previous_chunk_id,
                ChunkNeighbor(
                    chunk_id=neighbor.previous_chunk_id,
                ),
            )
            updated_previous = ChunkNeighbor(
                chunk_id=previous.chunk_id,
                previous_chunk_id=previous.previous_chunk_id,
                next_chunk_id=neighbor.chunk_id,
            )
            self._neighbors[previous.chunk_id] = updated_previous
            self._update_chunk_neighbor_metadata(updated_previous)

        if neighbor.next_chunk_id is not None:
            next_neighbor = self._neighbors.get(
                neighbor.next_chunk_id,
                ChunkNeighbor(
                    chunk_id=neighbor.next_chunk_id,
                ),
            )
            updated_next = ChunkNeighbor(
                chunk_id=next_neighbor.chunk_id,
                previous_chunk_id=neighbor.chunk_id,
                next_chunk_id=next_neighbor.next_chunk_id,
            )
            self._neighbors[next_neighbor.chunk_id] = updated_next
            self._update_chunk_neighbor_metadata(updated_next)

    def _update_chunk_neighbor_metadata(
        self,
        neighbor: ChunkNeighbor,
    ) -> None:
        chunk = self._chunks.get(neighbor.chunk_id)
        if chunk is None:
            return

        self._chunks[neighbor.chunk_id] = self._set_chunk_metadata_values(
            chunk,
            {
                "previous_chunk_id": neighbor.previous_chunk_id,
                "next_chunk_id": neighbor.next_chunk_id,
            },
        )

    @staticmethod
    def _filter_by_text(
        chunks: Iterable[Chunk],
        query: str,
    ) -> list[Chunk]:
        query_terms = {
            term
            for term in ChunkRepository._normalize_text(query).split()
            if term
        }

        if not query_terms:
            return list(chunks)

        scored_chunks: list[tuple[int, Chunk]] = []

        for chunk in chunks:
            text = ChunkRepository._normalize_text(
                ChunkRepository._get_chunk_text(chunk),
            )
            score = sum(
                text.count(term)
                for term in query_terms
            )

            if score > 0:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(
            key=lambda item: (
                -item[0],
                ChunkRepository._sort_key(item[1]),
            ),
        )

        return [chunk for _, chunk in scored_chunks]

    @staticmethod
    def _sort_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
        return sorted(
            chunks,
            key=ChunkRepository._sort_key,
        )

    @staticmethod
    def _sort_key(chunk: Chunk) -> tuple[Any, ...]:
        metadata = ChunkRepository._get_metadata(chunk)

        report_id = ChunkRepository._get_value(
            chunk,
            metadata,
            ("report_id",),
        )
        page_number = ChunkRepository._get_value(
            chunk,
            metadata,
            ("page_number", "source_page", "page"),
        )
        position = ChunkRepository._get_value(
            chunk,
            metadata,
            (
                "chunk_position",
                "chunk_index",
                "sequence_number",
                "order",
            ),
        )

        return (
            str(report_id or ""),
            ChunkRepository._to_int(page_number) or 0,
            ChunkRepository._to_int(position) or 0,
            ChunkRepository._get_chunk_id(chunk),
        )

    @staticmethod
    def _get_chunk_id(chunk: Chunk) -> str:
        # Chunk modelinin `id` alanı UUID olabilir. Retrieval, vector store ve
        # SQLite ise özgün chunk_uid ile çalıştığı için metadata kimliği
        # önceliklendirilir.
        metadata = ChunkRepository._get_metadata(chunk)

        for key in (
            "original_chunk_uid",
            "chunk_id",
            "identifier",
            "id",
        ):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()

        for attribute_name in ("chunk_id", "identifier", "id"):
            value = getattr(chunk, attribute_name, None)
            if value is not None and str(value).strip():
                return str(value).strip()

        raise ChunkRepositoryError(
            "Chunk üzerinde chunk_id, id veya identifier alanı bulunamadı.",
        )

    @staticmethod
    def _get_chunk_text(chunk: Chunk) -> str:
        for attribute_name in ("text", "content", "chunk_text"):
            value = getattr(chunk, attribute_name, None)
            if isinstance(value, str):
                return value

        return ""

    @staticmethod
    def _get_metadata(chunk: Chunk) -> dict[str, Any]:
        metadata = getattr(chunk, "metadata", None)

        if metadata is None:
            return {}

        if isinstance(metadata, Mapping):
            return dict(metadata)

        if hasattr(metadata, "model_dump"):
            return dict(metadata.model_dump(exclude_none=True))

        if hasattr(metadata, "dict"):
            return dict(metadata.dict(exclude_none=True))

        return {}

    @staticmethod
    def _get_value(
        chunk: Chunk,
        metadata: Mapping[str, Any],
        keys: Sequence[str],
    ) -> Any:
        for key in keys:
            # Chunk modelindeki report_id UUID'dir; uygulamanın retrieval
            # filtresi ise report_uid metnini kullanır.
            if key == "report_id":
                metadata_value = (
                    metadata.get("original_report_uid")
                    or metadata.get("report_id")
                )
                if metadata_value is not None:
                    return getattr(
                        metadata_value,
                        "value",
                        metadata_value,
                    )

            value = getattr(chunk, key, None)
            if value is not None:
                return getattr(value, "value", value)

            value = metadata.get(key)
            if value is not None:
                return getattr(value, "value", value)

        return None

    @staticmethod
    def _set_chunk_metadata_values(
        chunk: Chunk,
        values: Mapping[str, Any],
    ) -> Chunk:
        metadata = ChunkRepository._get_metadata(chunk)
        metadata.update(values)

        if hasattr(chunk, "model_copy"):
            return chunk.model_copy(
                update={"metadata": metadata},
                deep=True,
            )

        if hasattr(chunk, "copy"):
            try:
                return chunk.copy(
                    update={"metadata": metadata},
                    deep=True,
                )
            except (TypeError, ValueError):
                pass

        if is_dataclass(chunk):
            field_names = {item.name for item in fields(chunk)}
            if "metadata" not in field_names:
                raise ChunkRepositoryError(
                    "Chunk dataclass modelinde metadata alanı bulunamadı.",
                )
            try:
                return replace(chunk, metadata=metadata)
            except (TypeError, ValueError) as exc:
                raise ChunkRepositoryError(
                    "Dataclass chunk metadata alanı güncellenemedi.",
                ) from exc

        copied_chunk = deepcopy(chunk)
        setattr(copied_chunk, "metadata", metadata)
        return copied_chunk

    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
        if hasattr(chunk, "model_dump"):
            return dict(
                chunk.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
            )

        if hasattr(chunk, "dict"):
            return dict(chunk.dict(exclude_none=True))

        if is_dataclass(chunk):
            return asdict(chunk)

        if hasattr(chunk, "__dict__"):
            return {
                key: value
                for key, value in vars(chunk).items()
                if not key.startswith("_")
            }

        raise ChunkRepositoryError(
            "Chunk JSON formatına dönüştürülemedi.",
        )

    @staticmethod
    def _chunk_from_dict(data: Any) -> Chunk:
        if not isinstance(data, Mapping):
            raise ChunkRepositoryError(
                "Chunk kaydı mapping türünde olmalıdır.",
            )

        if hasattr(Chunk, "model_validate"):
            return Chunk.model_validate(data)

        if hasattr(Chunk, "parse_obj"):
            return Chunk.parse_obj(data)

        try:
            return Chunk(**dict(data))
        except TypeError as exc:
            raise ChunkRepositoryError(
                "Chunk kaydı model nesnesine dönüştürülemedi.",
            ) from exc

    @staticmethod
    def _normalize_identifier(value: Any) -> str:
        if value is None or not str(value).strip():
            raise ValueError("Chunk kimliği boş olamaz.")

        return str(value).strip()

    @staticmethod
    def _normalize_optional_identifier(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""

        enum_value = getattr(value, "value", value)

        return " ".join(
            str(enum_value).casefold().split(),
        )

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_sequence(value: Any) -> list[Any]:
        if value is None:
            return []

        if isinstance(value, (str, bytes)):
            return [value]

        if isinstance(value, Sequence):
            return list(value)

        if isinstance(value, set):
            return list(value)

        return [value]

    @classmethod
    def _as_normalized_set(
        cls,
        value: Any,
    ) -> set[str]:
        return {
            cls._normalize_text(item)
            for item in cls._as_sequence(value)
            if cls._normalize_text(item)
        }

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)

        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return enum_value

        if is_dataclass(value):
            return asdict(value)

        if hasattr(value, "model_dump"):
            return value.model_dump(
                mode="json",
                exclude_none=True,
            )

        raise TypeError(
            f"{type(value).__name__} JSON formatına dönüştürülemedi.",
        )


__all__ = [
    "ChunkNeighbor",
    "ChunkNotFoundError",
    "ChunkRepository",
    "ChunkRepositoryError",
    "ChunkSearchFilter",
    "DuplicateChunkError",
]