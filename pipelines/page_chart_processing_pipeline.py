from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from storage.database import Database
from table_processing.vlm_table_fallback import (
    CloudflareQuotaExceededError,
    VLMTableFallback,
)


@dataclass(slots=True, frozen=True)
class PageChartProcessingRequest:
    report_id: str
    max_candidates: int = 0
    resume: bool = True
    retry_failed: bool = False
    retry_no_points: bool = False
    retry_invalid: bool = False
    wait_between_calls_seconds: float = 4.0
    fallback_rate_limit_wait_seconds: float = 65.0
    stop_on_rate_limit: bool = False


@dataclass(slots=True, frozen=True)
class PageChartProcessingResult:
    report_id: str
    discovered_count: int
    pending_count: int
    attempted_count: int
    completed_count: int
    no_points_count: int
    invalid_count: int
    failed_count: int
    rate_limit_count: int
    skipped_count: int
    state_path: Path
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state_path"] = str(self.state_path)
        return result


class PageChartProcessingPipeline:
    """
    Ingestion tarafından üretilmiş grafik kırpımlarını bağımsız olarak işler.

    Bu pipeline PDF'yi yeniden indirmez, render etmez veya tabloları yeniden
    çıkarmaz. Yalnız mevcut `page_chart_candidates/*.png` dosyalarını Groq
    Vision'a gönderir ve başarılı grafik chunk'larını SQLite ile vector store'a
    yazar.
    """

    TERMINAL_STATUSES = {
        "completed",
        "no_points",
        "invalid",
    }

    def __init__(
        self,
        *,
        data_directory: str | Path = "data",
        database: Database | None = None,
        vector_store: Any | None = None,
        vlm_table_fallback: VLMTableFallback | None = None,
    ) -> None:
        self.data_directory = Path(
            data_directory
        ).expanduser().resolve()
        self.database = database or Database(
            database_path=self.data_directory / "sustainability.db"
        )
        self.vector_store = vector_store
        self.vlm = vlm_table_fallback or VLMTableFallback()

    def run(
        self,
        request: PageChartProcessingRequest,
    ) -> PageChartProcessingResult:
        report = self.database.get_report_by_uid(
            request.report_id
        )
        if report is None:
            raise ValueError(
                f"Rapor veritabanında bulunamadı: {request.report_id}"
            )

        report_database_id = int(report["id"])
        candidate_root = (
            self.data_directory
            / "table_images"
            / request.report_id
            / "page_chart_candidates"
        )
        if not candidate_root.is_dir():
            raise FileNotFoundError(
                "Grafik aday klasörü bulunamadı: "
                f"{candidate_root}"
            )

        state_root = (
            self.data_directory
            / "table_images"
            / request.report_id
            / "page_chart_state"
        )
        state_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        state_path = state_root / "candidate_state.json"
        state = self._load_state(state_path)

        image_paths = sorted(
            [
                *candidate_root.glob("*.png"),
                *candidate_root.glob("*.jpg"),
                *candidate_root.glob("*.jpeg"),
                *candidate_root.glob("*.webp"),
            ],
            key=self._candidate_sort_key,
        )

        pending_paths = [
            path
            for path in image_paths
            if self._should_process(
                path=path,
                state=state,
                request=request,
            )
        ]

        if request.max_candidates > 0:
            pending_paths = pending_paths[
                : request.max_candidates
            ]

        counters = {
            "attempted": 0,
            "completed": 0,
            "no_points": 0,
            "invalid": 0,
            "failed": 0,
            "rate_limit": 0,
            "skipped": 0,
        }
        errors: list[str] = []
        last_call_finished_at: float | None = None

        for position, image_path in enumerate(
            pending_paths,
            start=1,
        ):
            candidate_id = image_path.stem
            page_number = self._extract_page_number(
                image_path
            )
            page_record = self.database.fetch_one(
                """
                SELECT id, text_content
                FROM pages
                WHERE report_id = ?
                  AND page_number = ?
                """,
                (
                    report_database_id,
                    page_number,
                ),
            )
            page_id = (
                int(page_record["id"])
                if page_record is not None
                else None
            )
            page_text = (
                str(page_record.get("text_content") or "")
                if page_record is not None
                else ""
            )

            if last_call_finished_at is not None:
                elapsed = (
                    time.monotonic()
                    - last_call_finished_at
                )
                remaining = (
                    request.wait_between_calls_seconds
                    - elapsed
                )
                if remaining > 0:
                    time.sleep(remaining)

            print(
                f"[{position}/{len(pending_paths)}] "
                f"Sayfa {page_number}, aday={candidate_id}",
                flush=True,
            )

            pseudo_visual = SimpleNamespace(
                table_id=candidate_id,
                page_index=max(page_number - 1, 0),
                page_number=page_number,
                table_index=0,
                table_image_path=image_path,
                image_path=image_path,
                visual_region_type="chart",
                metadata={
                    "page_chart_candidate": True,
                    "text_preview": page_text,
                },
            )

            counters["attempted"] += 1

            try:
                chart = self.vlm.extract_chart(
                    visual_table=pseudo_visual,
                    contextual_text=page_text or None,
                )
                last_call_finished_at = time.monotonic()

            except CloudflareQuotaExceededError as exc:
                last_call_finished_at = time.monotonic()
                counters["rate_limit"] += 1

                retry_after = getattr(
                    exc,
                    "retry_after_seconds",
                    None,
                )
                wait_seconds = (
                    float(retry_after)
                    if retry_after is not None
                    else request.fallback_rate_limit_wait_seconds
                )
                wait_seconds = max(wait_seconds, 1.0)

                state[candidate_id] = {
                    "status": "rate_limited",
                    "page_number": page_number,
                    "image_path": str(image_path),
                    "reason": str(exc),
                    "retry_after_seconds": wait_seconds,
                    "updated_at_epoch": time.time(),
                }
                self._save_state(
                    state_path,
                    state,
                )

                print(
                    "Groq rate limit: "
                    f"{wait_seconds:.1f} saniye beklenecek.",
                    flush=True,
                )

                if request.stop_on_rate_limit:
                    break

                time.sleep(wait_seconds)
                continue

            except Exception as exc:
                last_call_finished_at = time.monotonic()
                counters["failed"] += 1
                message = (
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(
                    f"{candidate_id}: {message}"
                )
                state[candidate_id] = {
                    "status": "failed",
                    "page_number": page_number,
                    "image_path": str(image_path),
                    "reason": message,
                    "updated_at_epoch": time.time(),
                }
                self._save_state(
                    state_path,
                    state,
                )
                continue

            validation_reason = self._validation_reason(
                chart
            )
            points = tuple(
                getattr(chart, "points", ()) or ()
            )

            if not bool(
                getattr(chart, "is_valid", False)
            ):
                status = (
                    "no_points"
                    if validation_reason == "no_points"
                    or not points
                    else "invalid"
                )
                counters[status] += 1
                state[candidate_id] = {
                    "status": status,
                    "page_number": page_number,
                    "image_path": str(image_path),
                    "reason": validation_reason,
                    "point_count": len(points),
                    "confidence": float(
                        getattr(
                            chart,
                            "overall_confidence",
                            0.0,
                        )
                        or 0.0
                    ),
                    "chart": self._serialize_chart(
                        chart
                    ),
                    "updated_at_epoch": time.time(),
                }
                self._save_state(
                    state_path,
                    state,
                )
                print(
                    f"  -> {status}: "
                    f"points={len(points)}",
                    flush=True,
                )
                continue

            chunk_record = self._build_chunk_record(
                chart=chart,
                report_uid=request.report_id,
                report_database_id=report_database_id,
                page_id=page_id,
                page_number=page_number,
                candidate_id=candidate_id,
            )

            self.database.upsert_chunk(
                chunk_uid=chunk_record["chunk_uid"],
                report_id=report_database_id,
                page_id=page_id,
                page_number=page_number,
                chunk_index=chunk_record["chunk_index"],
                section_title=chunk_record[
                    "section_title"
                ],
                subsection_title=None,
                text_content=chunk_record[
                    "text_content"
                ],
                token_count=None,
                category=chunk_record["category"],
                language="en",
                previous_chunk_uid=None,
                next_chunk_uid=None,
                content_hash=chunk_record[
                    "content_hash"
                ],
                metadata=chunk_record["metadata"],
            )

            self._index_vector(
                chunk_record
            )

            counters["completed"] += 1
            state[candidate_id] = {
                "status": "completed",
                "page_number": page_number,
                "image_path": str(image_path),
                "chunk_uid": chunk_record[
                    "chunk_uid"
                ],
                "content_hash": chunk_record[
                    "content_hash"
                ],
                "confidence": chunk_record[
                    "metadata"
                ]["chart_confidence"],
                "chart": self._serialize_chart(
                    chart
                ),
                "updated_at_epoch": time.time(),
            }
            self._save_state(
                state_path,
                state,
            )
            print(
                "  -> completed: "
                f"{chunk_record['chunk_uid']}",
                flush=True,
            )

        return PageChartProcessingResult(
            report_id=request.report_id,
            discovered_count=len(image_paths),
            pending_count=len(pending_paths),
            attempted_count=counters["attempted"],
            completed_count=counters["completed"],
            no_points_count=counters["no_points"],
            invalid_count=counters["invalid"],
            failed_count=counters["failed"],
            rate_limit_count=counters["rate_limit"],
            skipped_count=counters["skipped"],
            state_path=state_path,
            errors=tuple(errors),
        )

    def _should_process(
        self,
        *,
        path: Path,
        state: Mapping[str, Mapping[str, Any]],
        request: PageChartProcessingRequest,
    ) -> bool:
        if not request.resume:
            return True

        status = str(
            state.get(path.stem, {}).get("status")
            or ""
        ).casefold()

        if not status:
            return True
        if status == "completed":
            return False
        if status == "no_points":
            return request.retry_no_points
        if status == "invalid":
            return request.retry_invalid
        if status == "failed":
            return request.retry_failed
        if status == "rate_limited":
            return True
        return True

    def _build_chunk_record(
        self,
        *,
        chart: Any,
        report_uid: str,
        report_database_id: int,
        page_id: int | None,
        page_number: int,
        candidate_id: str,
    ) -> dict[str, Any]:
        matrix = tuple(
            getattr(chart, "matrix", ()) or ()
        )
        rows = [
            " | ".join(
                str(cell).strip()
                for cell in row
            )
            for row in matrix
            if any(
                str(cell).strip()
                for cell in row
            )
        ]
        if not rows:
            raise ValueError(
                "Geçerli grafik sonucu boş matrix taşıyor."
            )

        title = str(
            getattr(chart, "title", "")
            or "Visual chart"
        ).strip()
        text_content = "\n".join(
            [
                (
                    f"Arçelik 2023 sürdürülebilirlik "
                    f"raporu. Sayfa {page_number}. "
                    f"Tam sayfa grafik kaynağı: "
                    f"{candidate_id}."
                ),
                title,
                *rows,
            ]
        )
        normalized = re.sub(
            r"\s+",
            " ",
            text_content,
        ).strip()
        content_hash = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        category, topic = self._infer_category_topic(
            text_content
        )
        chunk_uid = (
            f"{report_uid}_layout_chart_"
            f"{candidate_id}"
        )

        # Grafik worker chunk'ları için ayrılmış, sayfadan bağımsız yüksek aralık.
        chunk_index = (
            1_300_000
            + int(
                hashlib.sha256(
                    candidate_id.encode("utf-8")
                ).hexdigest()[:8],
                16,
            )
            % 500_000
        )

        metadata = {
            "report_id": report_uid,
            "database_report_id": report_database_id,
            "page_id": page_id,
            "page_number": page_number,
            "source_element_id": candidate_id,
            "source_type": "layout_visual_chart",
            "chart_type": getattr(
                chart,
                "chart_type",
                None,
            ),
            "chart_title": title,
            "chart_unit": getattr(
                chart,
                "unit",
                None,
            ),
            "chart_years": list(
                getattr(chart, "years", ()) or ()
            ),
            "chart_series": list(
                getattr(
                    chart,
                    "series_names",
                    (),
                )
                or ()
            ),
            "chart_confidence": float(
                getattr(
                    chart,
                    "overall_confidence",
                    0.0,
                )
                or 0.0
            ),
            "category": category,
            "topic": topic,
            "topics": [topic] if topic else [],
            "is_visual_evidence": True,
            "is_chart_evidence": True,
            "chart_worker": True,
            "content_hash": content_hash,
        }

        return {
            "chunk_uid": chunk_uid,
            "chunk_index": chunk_index,
            "section_title": title,
            "text_content": text_content,
            "category": category,
            "content_hash": content_hash,
            "metadata": metadata,
        }

    def _index_vector(
        self,
        chunk_record: Mapping[str, Any],
    ) -> None:
        if self.vector_store is None:
            return

        add_text = getattr(
            self.vector_store,
            "add_text",
            None,
        )
        if callable(add_text):
            add_text(
                chunk_record["chunk_uid"],
                chunk_record["text_content"],
                metadata=dict(
                    chunk_record["metadata"]
                ),
                overwrite=True,
            )
            return

        add_texts = getattr(
            self.vector_store,
            "add_texts",
            None,
        )
        if callable(add_texts):
            add_texts(
                [chunk_record["text_content"]],
                document_ids=[
                    chunk_record["chunk_uid"]
                ],
                metadatas=[
                    dict(chunk_record["metadata"])
                ],
                overwrite=True,
            )

    @staticmethod
    def _infer_category_topic(
        text: str,
    ) -> tuple[str, str | None]:
        normalized = text.casefold()

        rules = (
            (
                "environmental",
                "water",
                (
                    "water",
                    "withdrawal",
                    "wastewater",
                    "recycled water",
                ),
            ),
            (
                "environmental",
                "energy",
                (
                    "energy",
                    "electricity",
                    "renewable",
                    "natural gas",
                ),
            ),
            (
                "environmental",
                "greenhouse_gas_emissions",
                (
                    "scope 1",
                    "scope 2",
                    "scope 3",
                    "co2",
                    "emission",
                ),
            ),
            (
                "environmental",
                "waste",
                (
                    "waste",
                    "landfill",
                    "recycling",
                ),
            ),
            (
                "social",
                "employees",
                (
                    "employee",
                    "workforce",
                    "female",
                    "male",
                ),
            ),
            (
                "social",
                "occupational_health_and_safety",
                (
                    "injury",
                    "accident",
                    "ltifr",
                    "trir",
                ),
            ),
        )

        for category, topic, terms in rules:
            if any(
                term in normalized
                for term in terms
            ):
                return category, topic

        return "general", None

    @staticmethod
    def _validation_reason(
        chart: Any,
    ) -> str:
        metadata = getattr(
            chart,
            "metadata",
            {},
        ) or {}
        if isinstance(metadata, Mapping):
            reason = metadata.get(
                "validation_reason"
            )
            if reason:
                return str(reason)

        points = tuple(
            getattr(chart, "points", ()) or ()
        )
        if not points:
            return "no_points"

        return "invalid"

    @staticmethod
    def _serialize_chart(
        chart: Any,
    ) -> dict[str, Any]:
        to_dict = getattr(
            chart,
            "to_dict",
            None,
        )
        if callable(to_dict):
            try:
                return dict(to_dict())
            except Exception:
                pass

        result: dict[str, Any] = {}
        for name in (
            "table_id",
            "page_index",
            "page_number",
            "title",
            "chart_type",
            "unit",
            "years",
            "series_names",
            "matrix",
            "overall_confidence",
            "warnings",
            "metadata",
        ):
            value = getattr(
                chart,
                name,
                None,
            )
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            result[name] = value
        return result

    @staticmethod
    def _load_state(
        path: Path,
    ) -> dict[str, dict[str, Any]]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return {}

        if not isinstance(payload, dict):
            return {}

        return {
            str(key): dict(value)
            for key, value in payload.items()
            if isinstance(value, Mapping)
        }

    @staticmethod
    def _save_state(
        path: Path,
        state: Mapping[str, Mapping[str, Any]],
    ) -> None:
        temporary_path = path.with_suffix(
            ".tmp"
        )
        temporary_path.write_text(
            json.dumps(
                state,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    @staticmethod
    def _extract_page_number(
        path: Path,
    ) -> int:
        match = re.search(
            r"page_(\d+)",
            path.stem,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(
                f"Dosya adından sayfa numarası alınamadı: {path.name}"
            )
        return int(match.group(1))

    @classmethod
    def _candidate_sort_key(
        cls,
        path: Path,
    ) -> tuple[int, str]:
        try:
            page_number = cls._extract_page_number(
                path
            )
        except ValueError:
            page_number = 10**9
        return page_number, path.name


__all__ = [
    "PageChartProcessingPipeline",
    "PageChartProcessingRequest",
    "PageChartProcessingResult",
]