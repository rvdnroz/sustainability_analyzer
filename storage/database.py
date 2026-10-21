from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Mapping, Sequence


logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """Veri tabanı işlemi başarısız olduğunda yükseltilir."""


class RecordNotFoundError(DatabaseError):
    """İstenen veri tabanı kaydı bulunamadığında yükseltilir."""


class Database:
    """
    Sustainability Analyzer için SQLite veri tabanı katmanı.

    Saklanan temel varlıklar:

    - reports
    - pages
    - tables
    - chunks
    - queries
    - evidence
    - answers

    Tasarım ilkeleri:

    - SQLite üzerinde hafif ve taşınabilir depolama
    - Foreign key bütünlüğü
    - JSON metadata alanları
    - Tekrar çalıştırılabilir upsert işlemleri
    - Rapor, sayfa ve soru bazlı izlenebilirlik
    - Thread başına ayrı bağlantı
    - Transaction desteği
    """

    def __init__(
        self,
        database_path: str | Path = "data/sustainability_analyzer.db",
        *,
        timeout_seconds: float = 30.0,
        enable_wal: bool = True,
        initialize_schema: bool = True,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds pozitif olmalıdır."
            )

        self.database_path = Path(
            database_path
        ).expanduser().resolve()

        self.timeout_seconds = timeout_seconds
        self.enable_wal = enable_wal

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._local = threading.local()

        if initialize_schema:
            self.initialize()

    def initialize(self) -> None:
        """Veri tabanı şemasını ve indeksleri oluşturur."""

        schema = """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            report_uid TEXT NOT NULL UNIQUE,
            company_name TEXT NOT NULL,
            company_slug TEXT NOT NULL,
            report_year INTEGER NOT NULL,

            report_type TEXT,
            source_url TEXT,
            final_url TEXT,
            local_pdf_path TEXT,

            sha256 TEXT,
            file_size_bytes INTEGER,
            page_count INTEGER,

            status TEXT NOT NULL DEFAULT 'created',
            validation_status TEXT,
            validation_message TEXT,

            metadata_json TEXT NOT NULL DEFAULT '{}',

            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            UNIQUE(company_slug, report_year, report_uid)
        );

        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            page_uid TEXT NOT NULL UNIQUE,
            report_id INTEGER NOT NULL,

            page_index INTEGER NOT NULL,
            page_number INTEGER NOT NULL,

            page_type TEXT,
            classification_confidence REAL,

            width_points REAL,
            height_points REAL,
            rotation INTEGER,

            has_native_text INTEGER NOT NULL DEFAULT 0,
            needs_ocr INTEGER NOT NULL DEFAULT 0,
            is_scanned INTEGER NOT NULL DEFAULT 0,

            text_content TEXT,
            rendered_image_path TEXT,

            classification_json TEXT NOT NULL DEFAULT '{}',
            layout_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',

            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            FOREIGN KEY(report_id)
                REFERENCES reports(id)
                ON DELETE CASCADE,

            UNIQUE(report_id, page_number)
        );

        CREATE TABLE IF NOT EXISTS tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            table_uid TEXT NOT NULL UNIQUE,
            report_id INTEGER NOT NULL,
            page_id INTEGER,

            page_number INTEGER NOT NULL,
            table_index INTEGER NOT NULL,

            title TEXT,
            caption TEXT,

            extraction_method TEXT,
            confidence REAL,

            row_count INTEGER,
            column_count INTEGER,

            bounding_box_json TEXT NOT NULL DEFAULT '{}',
            headers_json TEXT NOT NULL DEFAULT '[]',
            rows_json TEXT NOT NULL DEFAULT '[]',
            normalized_data_json TEXT NOT NULL DEFAULT '{}',
            validation_issues_json TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}',

            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            FOREIGN KEY(report_id)
                REFERENCES reports(id)
                ON DELETE CASCADE,

            FOREIGN KEY(page_id)
                REFERENCES pages(id)
                ON DELETE SET NULL,

            UNIQUE(report_id, page_number, table_index)
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            chunk_uid TEXT NOT NULL UNIQUE,
            report_id INTEGER NOT NULL,
            page_id INTEGER,

            page_number INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,

            section_title TEXT,
            subsection_title TEXT,

            text_content TEXT NOT NULL,
            token_count INTEGER,

            category TEXT,
            language TEXT,

            previous_chunk_uid TEXT,
            next_chunk_uid TEXT,

            content_hash TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',

            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            FOREIGN KEY(report_id)
                REFERENCES reports(id)
                ON DELETE CASCADE,

            FOREIGN KEY(page_id)
                REFERENCES pages(id)
                ON DELETE SET NULL,

            UNIQUE(report_id, page_number, chunk_index)
        );

        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            query_uid TEXT NOT NULL UNIQUE,
            report_id INTEGER,

            question_id TEXT,
            question_text TEXT NOT NULL,

            intent TEXT,
            answer_type TEXT,
            category TEXT,
            target_year INTEGER,

            retrieval_route TEXT,
            expanded_queries_json TEXT NOT NULL DEFAULT '[]',
            filters_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',

            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            FOREIGN KEY(report_id)
                REFERENCES reports(id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            evidence_uid TEXT NOT NULL UNIQUE,
            query_id INTEGER NOT NULL,
            report_id INTEGER NOT NULL,

            source_type TEXT NOT NULL,
            source_uid TEXT,

            page_number INTEGER,
            table_id INTEGER,
            chunk_id INTEGER,

            text_content TEXT,

            lexical_score REAL,
            vector_score REAL,
            reranker_score REAL,
            fusion_score REAL,
            final_score REAL,

            rank_position INTEGER,
            is_valid INTEGER NOT NULL DEFAULT 1,

            validation_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',

            created_at TEXT NOT NULL,

            FOREIGN KEY(query_id)
                REFERENCES queries(id)
                ON DELETE CASCADE,

            FOREIGN KEY(report_id)
                REFERENCES reports(id)
                ON DELETE CASCADE,

            FOREIGN KEY(table_id)
                REFERENCES tables(id)
                ON DELETE SET NULL,

            FOREIGN KEY(chunk_id)
                REFERENCES chunks(id)
                ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            answer_uid TEXT NOT NULL UNIQUE,
            query_id INTEGER NOT NULL,
            report_id INTEGER NOT NULL,

            answer_text TEXT,
            answer_status TEXT NOT NULL,

            confidence REAL,
            model_name TEXT,

            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,

            evidence_uids_json TEXT NOT NULL DEFAULT '[]',
            citations_json TEXT NOT NULL DEFAULT '[]',
            raw_response_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',

            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            FOREIGN KEY(query_id)
                REFERENCES queries(id)
                ON DELETE CASCADE,

            FOREIGN KEY(report_id)
                REFERENCES reports(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_reports_company_year
            ON reports(company_slug, report_year);

        CREATE INDEX IF NOT EXISTS idx_reports_sha256
            ON reports(sha256);

        CREATE INDEX IF NOT EXISTS idx_pages_report
            ON pages(report_id);

        CREATE INDEX IF NOT EXISTS idx_pages_report_number
            ON pages(report_id, page_number);

        CREATE INDEX IF NOT EXISTS idx_pages_type
            ON pages(page_type);

        CREATE INDEX IF NOT EXISTS idx_tables_report
            ON tables(report_id);

        CREATE INDEX IF NOT EXISTS idx_tables_page
            ON tables(page_id);

        CREATE INDEX IF NOT EXISTS idx_chunks_report
            ON chunks(report_id);

        CREATE INDEX IF NOT EXISTS idx_chunks_page
            ON chunks(page_id);

        CREATE INDEX IF NOT EXISTS idx_chunks_hash
            ON chunks(content_hash);

        CREATE INDEX IF NOT EXISTS idx_queries_report
            ON queries(report_id);

        CREATE INDEX IF NOT EXISTS idx_queries_question_id
            ON queries(question_id);

        CREATE INDEX IF NOT EXISTS idx_evidence_query
            ON evidence(query_id);

        CREATE INDEX IF NOT EXISTS idx_evidence_report
            ON evidence(report_id);

        CREATE INDEX IF NOT EXISTS idx_evidence_score
            ON evidence(final_score DESC);

        CREATE INDEX IF NOT EXISTS idx_answers_query
            ON answers(query_id);

        CREATE INDEX IF NOT EXISTS idx_answers_report
            ON answers(report_id);
        """

        try:
            with self.connection() as connection:
                connection.executescript(schema)

            logger.info(
                "Veri tabanı şeması hazırlandı: %s",
                self.database_path,
            )

        except sqlite3.Error as exc:
            raise DatabaseError(
                "Veri tabanı şeması oluşturulamadı."
            ) from exc

    def get_connection(self) -> sqlite3.Connection:
        """
        Aktif thread için SQLite bağlantısı döndürür.

        Her thread kendi bağlantısını kullanır.
        """

        connection = getattr(
            self._local,
            "connection",
            None,
        )

        if connection is None:
            try:
                connection = sqlite3.connect(
                    self.database_path,
                    timeout=self.timeout_seconds,
                    detect_types=sqlite3.PARSE_DECLTYPES,
                    check_same_thread=False,
                )

                connection.row_factory = sqlite3.Row

                connection.execute(
                    "PRAGMA foreign_keys = ON;"
                )

                connection.execute(
                    "PRAGMA busy_timeout = 30000;"
                )

                if self.enable_wal:
                    connection.execute(
                        "PRAGMA journal_mode = WAL;"
                    )

                    connection.execute(
                        "PRAGMA synchronous = NORMAL;"
                    )

                self._local.connection = connection

            except sqlite3.Error as exc:
                raise DatabaseError(
                    "Veri tabanı bağlantısı açılamadı."
                ) from exc

        return connection

    @contextmanager
    def connection(
        self,
    ) -> Generator[sqlite3.Connection, None, None]:
        """
        Transaction kontrollü bağlantı context manager'ı.

        Başarı durumunda commit, hata durumunda rollback yapılır.
        """

        connection = self.get_connection()

        try:
            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

    def close(self) -> None:
        """Aktif thread bağlantısını kapatır."""

        connection = getattr(
            self._local,
            "connection",
            None,
        )

        if connection is None:
            return

        try:
            connection.close()

        except sqlite3.Error as exc:
            raise DatabaseError(
                "Veri tabanı bağlantısı kapatılamadı."
            ) from exc

        finally:
            self._local.connection = None

    def execute(
        self,
        sql: str,
        parameters: Sequence[Any] | Mapping[str, Any] = (),
    ) -> sqlite3.Cursor:
        """Tek bir SQL sorgusu çalıştırır."""

        try:
            with self.connection() as connection:
                cursor = connection.execute(
                    sql,
                    parameters,
                )

                return cursor

        except sqlite3.Error as exc:
            raise DatabaseError(
                f"SQL sorgusu çalıştırılamadı: {sql}"
            ) from exc

    def fetch_one(
        self,
        sql: str,
        parameters: Sequence[Any] | Mapping[str, Any] = (),
    ) -> dict[str, Any] | None:
        """Tek kayıt döndürür."""

        try:
            connection = self.get_connection()

            cursor = connection.execute(
                sql,
                parameters,
            )

            row = cursor.fetchone()

            return (
                self._row_to_dict(row)
                if row is not None
                else None
            )

        except sqlite3.Error as exc:
            raise DatabaseError(
                "Veri tabanı kaydı okunamadı."
            ) from exc

    def fetch_all(
        self,
        sql: str,
        parameters: Sequence[Any] | Mapping[str, Any] = (),
    ) -> list[dict[str, Any]]:
        """Sorgunun bütün kayıtlarını döndürür."""

        try:
            connection = self.get_connection()

            cursor = connection.execute(
                sql,
                parameters,
            )

            return [
                self._row_to_dict(row)
                for row in cursor.fetchall()
            ]

        except sqlite3.Error as exc:
            raise DatabaseError(
                "Veri tabanı kayıtları okunamadı."
            ) from exc

    def upsert_report(
        self,
        *,
        report_uid: str,
        company_name: str,
        company_slug: str,
        report_year: int,
        report_type: str | None = None,
        source_url: str | None = None,
        final_url: str | None = None,
        local_pdf_path: str | Path | None = None,
        sha256: str | None = None,
        file_size_bytes: int | None = None,
        page_count: int | None = None,
        status: str = "created",
        validation_status: str | None = None,
        validation_message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Rapor kaydı oluşturur veya günceller."""

        self._validate_non_empty(
            report_uid,
            "report_uid",
        )

        self._validate_non_empty(
            company_name,
            "company_name",
        )

        self._validate_non_empty(
            company_slug,
            "company_slug",
        )

        self._validate_year(
            report_year
        )

        now = self._utc_now()

        sql = """
        INSERT INTO reports (
            report_uid,
            company_name,
            company_slug,
            report_year,
            report_type,
            source_url,
            final_url,
            local_pdf_path,
            sha256,
            file_size_bytes,
            page_count,
            status,
            validation_status,
            validation_message,
            metadata_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(report_uid)
        DO UPDATE SET
            company_name = excluded.company_name,
            company_slug = excluded.company_slug,
            report_year = excluded.report_year,
            report_type = excluded.report_type,
            source_url = excluded.source_url,
            final_url = excluded.final_url,
            local_pdf_path = excluded.local_pdf_path,
            sha256 = excluded.sha256,
            file_size_bytes = excluded.file_size_bytes,
            page_count = excluded.page_count,
            status = excluded.status,
            validation_status = excluded.validation_status,
            validation_message = excluded.validation_message,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at;
        """

        parameters = (
            report_uid,
            company_name,
            company_slug,
            report_year,
            report_type,
            source_url,
            final_url,
            self._path_to_string(local_pdf_path),
            sha256,
            file_size_bytes,
            page_count,
            status,
            validation_status,
            validation_message,
            self._to_json(metadata or {}),
            now,
            now,
        )

        try:
            with self.connection() as connection:
                connection.execute(
                    sql,
                    parameters,
                )

                row = connection.execute(
                    """
                    SELECT id
                    FROM reports
                    WHERE report_uid = ?
                    """,
                    (report_uid,),
                ).fetchone()

                if row is None:
                    raise DatabaseError(
                        "Rapor kaydı oluşturuldu ancak ID alınamadı."
                    )

                return int(row["id"])

        except sqlite3.Error as exc:
            raise DatabaseError(
                "Rapor kaydı oluşturulamadı."
            ) from exc

    def get_report(
        self,
        report_id: int,
    ) -> dict[str, Any]:
        """Rapor kaydını ID ile getirir."""

        row = self.fetch_one(
            """
            SELECT *
            FROM reports
            WHERE id = ?
            """,
            (report_id,),
        )

        if row is None:
            raise RecordNotFoundError(
                f"Rapor bulunamadı: {report_id}"
            )

        return self._deserialize_record(
            row
        )

    def get_report_by_uid(
        self,
        report_uid: str,
    ) -> dict[str, Any] | None:
        """Rapor kaydını report_uid ile getirir."""

        row = self.fetch_one(
            """
            SELECT *
            FROM reports
            WHERE report_uid = ?
            """,
            (report_uid,),
        )

        return (
            self._deserialize_record(row)
            if row is not None
            else None
        )

    def find_report(
        self,
        *,
        company_slug: str,
        report_year: int,
    ) -> list[dict[str, Any]]:
        """Şirket slug ve yıl ile raporları getirir."""

        rows = self.fetch_all(
            """
            SELECT *
            FROM reports
            WHERE company_slug = ?
              AND report_year = ?
            ORDER BY updated_at DESC
            """,
            (
                company_slug,
                report_year,
            ),
        )

        return [
            self._deserialize_record(row)
            for row in rows
        ]

    def update_report_status(
        self,
        report_id: int,
        status: str,
        *,
        validation_status: str | None = None,
        validation_message: str | None = None,
    ) -> None:
        """Rapor yaşam döngüsü durumunu günceller."""

        self._validate_non_empty(
            status,
            "status",
        )

        sql = """
        UPDATE reports
        SET
            status = ?,
            validation_status = COALESCE(?, validation_status),
            validation_message = COALESCE(?, validation_message),
            updated_at = ?
        WHERE id = ?
        """

        with self.connection() as connection:
            cursor = connection.execute(
                sql,
                (
                    status,
                    validation_status,
                    validation_message,
                    self._utc_now(),
                    report_id,
                ),
            )

            if cursor.rowcount == 0:
                raise RecordNotFoundError(
                    f"Rapor bulunamadı: {report_id}"
                )

    def upsert_page(
        self,
        *,
        page_uid: str,
        report_id: int,
        page_index: int,
        page_number: int,
        page_type: str | None = None,
        classification_confidence: float | None = None,
        width_points: float | None = None,
        height_points: float | None = None,
        rotation: int | None = None,
        has_native_text: bool = False,
        needs_ocr: bool = False,
        is_scanned: bool = False,
        text_content: str | None = None,
        rendered_image_path: str | Path | None = None,
        classification: Mapping[str, Any] | None = None,
        layout: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Sayfa kaydı oluşturur veya günceller."""

        if page_index < 0:
            raise ValueError(
                "page_index negatif olamaz."
            )

        if page_number < 1:
            raise ValueError(
                "page_number 1 veya daha büyük olmalıdır."
            )

        now = self._utc_now()

        sql = """
        INSERT INTO pages (
            page_uid,
            report_id,
            page_index,
            page_number,
            page_type,
            classification_confidence,
            width_points,
            height_points,
            rotation,
            has_native_text,
            needs_ocr,
            is_scanned,
            text_content,
            rendered_image_path,
            classification_json,
            layout_json,
            metadata_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(page_uid)
        DO UPDATE SET
            report_id = excluded.report_id,
            page_index = excluded.page_index,
            page_number = excluded.page_number,
            page_type = excluded.page_type,
            classification_confidence =
                excluded.classification_confidence,
            width_points = excluded.width_points,
            height_points = excluded.height_points,
            rotation = excluded.rotation,
            has_native_text = excluded.has_native_text,
            needs_ocr = excluded.needs_ocr,
            is_scanned = excluded.is_scanned,
            text_content = excluded.text_content,
            rendered_image_path = excluded.rendered_image_path,
            classification_json = excluded.classification_json,
            layout_json = excluded.layout_json,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at;
        """

        parameters = (
            page_uid,
            report_id,
            page_index,
            page_number,
            page_type,
            classification_confidence,
            width_points,
            height_points,
            rotation,
            int(has_native_text),
            int(needs_ocr),
            int(is_scanned),
            text_content,
            self._path_to_string(
                rendered_image_path
            ),
            self._to_json(
                classification or {}
            ),
            self._to_json(
                layout or {}
            ),
            self._to_json(
                metadata or {}
            ),
            now,
            now,
        )

        try:
            with self.connection() as connection:
                connection.execute(
                    sql,
                    parameters,
                )

                row = connection.execute(
                    """
                    SELECT id
                    FROM pages
                    WHERE page_uid = ?
                    """,
                    (page_uid,),
                ).fetchone()

                if row is None:
                    raise DatabaseError(
                        "Sayfa kaydı oluşturuldu ancak ID alınamadı."
                    )

                return int(row["id"])

        except sqlite3.Error as exc:
            raise DatabaseError(
                "Sayfa kaydı oluşturulamadı."
            ) from exc

    def get_pages(
        self,
        report_id: int,
    ) -> list[dict[str, Any]]:
        """Bir rapora ait bütün sayfaları getirir."""

        rows = self.fetch_all(
            """
            SELECT *
            FROM pages
            WHERE report_id = ?
            ORDER BY page_number ASC
            """,
            (report_id,),
        )

        return [
            self._deserialize_record(row)
            for row in rows
        ]

    def get_page(
        self,
        report_id: int,
        page_number: int,
    ) -> dict[str, Any] | None:
        """Belirli bir rapor sayfasını getirir."""

        row = self.fetch_one(
            """
            SELECT *
            FROM pages
            WHERE report_id = ?
              AND page_number = ?
            """,
            (
                report_id,
                page_number,
            ),
        )

        return (
            self._deserialize_record(row)
            if row is not None
            else None
        )

    def upsert_table(
        self,
        *,
        table_uid: str,
        report_id: int,
        page_number: int,
        table_index: int,
        page_id: int | None = None,
        title: str | None = None,
        caption: str | None = None,
        extraction_method: str | None = None,
        confidence: float | None = None,
        row_count: int | None = None,
        column_count: int | None = None,
        bounding_box: Mapping[str, Any] | Sequence[Any] | None = None,
        headers: Sequence[Any] | None = None,
        rows: Sequence[Any] | None = None,
        normalized_data: Mapping[str, Any] | Sequence[Any] | None = None,
        validation_issues: Sequence[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Tablo kaydı oluşturur veya günceller."""

        if page_number < 1:
            raise ValueError(
                "page_number 1 veya daha büyük olmalıdır."
            )

        if table_index < 0:
            raise ValueError(
                "table_index negatif olamaz."
            )

        now = self._utc_now()

        sql = """
        INSERT INTO tables (
            table_uid,
            report_id,
            page_id,
            page_number,
            table_index,
            title,
            caption,
            extraction_method,
            confidence,
            row_count,
            column_count,
            bounding_box_json,
            headers_json,
            rows_json,
            normalized_data_json,
            validation_issues_json,
            metadata_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(report_id, page_number, table_index)
        DO UPDATE SET
            table_uid = excluded.table_uid,
            page_id = excluded.page_id,
            title = excluded.title,
            caption = excluded.caption,
            extraction_method = excluded.extraction_method,
            confidence = excluded.confidence,
            row_count = excluded.row_count,
            column_count = excluded.column_count,
            bounding_box_json = excluded.bounding_box_json,
            headers_json = excluded.headers_json,
            rows_json = excluded.rows_json,
            normalized_data_json = excluded.normalized_data_json,
            validation_issues_json = excluded.validation_issues_json,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at;
        """

        parameters = (
            table_uid,
            report_id,
            page_id,
            page_number,
            table_index,
            title,
            caption,
            extraction_method,
            confidence,
            row_count,
            column_count,
            self._to_json(
                bounding_box or {}
            ),
            self._to_json(
                headers or []
            ),
            self._to_json(
                rows or []
            ),
            self._to_json(
                normalized_data or {}
            ),
            self._to_json(
                validation_issues or []
            ),
            self._to_json(
                metadata or {}
            ),
            now,
            now,
        )

        try:
            with self.connection() as connection:
                connection.execute(
                    sql,
                    parameters,
                )

                row = connection.execute(
                    """
                    SELECT id
                    FROM tables
                    WHERE report_id = ?
                      AND page_number = ?
                      AND table_index = ?
                    """,
                    (
                        report_id,
                        page_number,
                        table_index,
                    ),
                ).fetchone()

                if row is None:
                    raise DatabaseError(
                        "Tablo oluşturuldu ancak ID alınamadı."
                    )

                return int(row["id"])

        except sqlite3.Error as exc:
            raise DatabaseError(
                "Tablo kaydı oluşturulamadı."
            ) from exc

    def get_tables(
        self,
        report_id: int,
        *,
        page_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Bir rapora ait tablo kayıtlarını getirir."""

        if page_number is None:
            sql = """
            SELECT *
            FROM tables
            WHERE report_id = ?
            ORDER BY page_number, table_index
            """

            parameters = (report_id,)

        else:
            sql = """
            SELECT *
            FROM tables
            WHERE report_id = ?
              AND page_number = ?
            ORDER BY table_index
            """

            parameters = (
                report_id,
                page_number,
            )

        rows = self.fetch_all(
            sql,
            parameters,
        )

        return [
            self._deserialize_record(row)
            for row in rows
        ]

    def upsert_chunk(
        self,
        *,
        chunk_uid: str,
        report_id: int,
        page_number: int,
        chunk_index: int,
        text_content: str,
        page_id: int | None = None,
        section_title: str | None = None,
        subsection_title: str | None = None,
        token_count: int | None = None,
        category: str | None = None,
        language: str | None = None,
        previous_chunk_uid: str | None = None,
        next_chunk_uid: str | None = None,
        content_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Metin chunk kaydı oluşturur veya günceller."""

        self._validate_non_empty(
            text_content,
            "text_content",
        )

        if page_number < 1:
            raise ValueError(
                "page_number 1 veya daha büyük olmalıdır."
            )

        if chunk_index < 0:
            raise ValueError(
                "chunk_index negatif olamaz."
            )

        now = self._utc_now()

        sql = """
        INSERT INTO chunks (
            chunk_uid,
            report_id,
            page_id,
            page_number,
            chunk_index,
            section_title,
            subsection_title,
            text_content,
            token_count,
            category,
            language,
            previous_chunk_uid,
            next_chunk_uid,
            content_hash,
            metadata_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(chunk_uid)
        DO UPDATE SET
            report_id = excluded.report_id,
            page_id = excluded.page_id,
            page_number = excluded.page_number,
            chunk_index = excluded.chunk_index,
            section_title = excluded.section_title,
            subsection_title = excluded.subsection_title,
            text_content = excluded.text_content,
            token_count = excluded.token_count,
            category = excluded.category,
            language = excluded.language,
            previous_chunk_uid = excluded.previous_chunk_uid,
            next_chunk_uid = excluded.next_chunk_uid,
            content_hash = excluded.content_hash,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at;
        """

        parameters = (
            chunk_uid,
            report_id,
            page_id,
            page_number,
            chunk_index,
            section_title,
            subsection_title,
            text_content,
            token_count,
            category,
            language,
            previous_chunk_uid,
            next_chunk_uid,
            content_hash,
            self._to_json(
                metadata or {}
            ),
            now,
            now,
        )

        return self._execute_upsert_and_get_id(
            sql=sql,
            parameters=parameters,
            table_name="chunks",
            uid_column="chunk_uid",
            uid_value=chunk_uid,
            record_name="Chunk",
        )

    def get_chunks(
        self,
        report_id: int,
        *,
        page_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Bir rapora ait chunk kayıtlarını getirir."""

        if page_number is None:
            sql = """
            SELECT *
            FROM chunks
            WHERE report_id = ?
            ORDER BY page_number, chunk_index
            """

            parameters = (report_id,)

        else:
            sql = """
            SELECT *
            FROM chunks
            WHERE report_id = ?
              AND page_number = ?
            ORDER BY chunk_index
            """

            parameters = (
                report_id,
                page_number,
            )

        rows = self.fetch_all(
            sql,
            parameters,
        )

        return [
            self._deserialize_record(row)
            for row in rows
        ]

    def upsert_query(
        self,
        *,
        query_uid: str,
        question_text: str,
        report_id: int | None = None,
        question_id: str | None = None,
        intent: str | None = None,
        answer_type: str | None = None,
        category: str | None = None,
        target_year: int | None = None,
        retrieval_route: str | None = None,
        expanded_queries: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Soru ve query analizi kaydı oluşturur veya günceller."""

        self._validate_non_empty(
            query_uid,
            "query_uid",
        )

        self._validate_non_empty(
            question_text,
            "question_text",
        )

        now = self._utc_now()

        sql = """
        INSERT INTO queries (
            query_uid,
            report_id,
            question_id,
            question_text,
            intent,
            answer_type,
            category,
            target_year,
            retrieval_route,
            expanded_queries_json,
            filters_json,
            metadata_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(query_uid)
        DO UPDATE SET
            report_id = excluded.report_id,
            question_id = excluded.question_id,
            question_text = excluded.question_text,
            intent = excluded.intent,
            answer_type = excluded.answer_type,
            category = excluded.category,
            target_year = excluded.target_year,
            retrieval_route = excluded.retrieval_route,
            expanded_queries_json = excluded.expanded_queries_json,
            filters_json = excluded.filters_json,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at;
        """

        parameters = (
            query_uid,
            report_id,
            question_id,
            question_text,
            intent,
            answer_type,
            category,
            target_year,
            retrieval_route,
            self._to_json(
                expanded_queries or []
            ),
            self._to_json(
                filters or {}
            ),
            self._to_json(
                metadata or {}
            ),
            now,
            now,
        )

        return self._execute_upsert_and_get_id(
            sql=sql,
            parameters=parameters,
            table_name="queries",
            uid_column="query_uid",
            uid_value=query_uid,
            record_name="Sorgu",
        )

    def insert_evidence(
        self,
        *,
        evidence_uid: str,
        query_id: int,
        report_id: int,
        source_type: str,
        source_uid: str | None = None,
        page_number: int | None = None,
        table_id: int | None = None,
        chunk_id: int | None = None,
        text_content: str | None = None,
        lexical_score: float | None = None,
        vector_score: float | None = None,
        reranker_score: float | None = None,
        fusion_score: float | None = None,
        final_score: float | None = None,
        rank_position: int | None = None,
        is_valid: bool = True,
        validation: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Tek bir retrieval kanıtı kaydeder."""

        now = self._utc_now()

        sql = """
        INSERT INTO evidence (
            evidence_uid,
            query_id,
            report_id,
            source_type,
            source_uid,
            page_number,
            table_id,
            chunk_id,
            text_content,
            lexical_score,
            vector_score,
            reranker_score,
            fusion_score,
            final_score,
            rank_position,
            is_valid,
            validation_json,
            metadata_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(evidence_uid)
        DO UPDATE SET
            query_id = excluded.query_id,
            report_id = excluded.report_id,
            source_type = excluded.source_type,
            source_uid = excluded.source_uid,
            page_number = excluded.page_number,
            table_id = excluded.table_id,
            chunk_id = excluded.chunk_id,
            text_content = excluded.text_content,
            lexical_score = excluded.lexical_score,
            vector_score = excluded.vector_score,
            reranker_score = excluded.reranker_score,
            fusion_score = excluded.fusion_score,
            final_score = excluded.final_score,
            rank_position = excluded.rank_position,
            is_valid = excluded.is_valid,
            validation_json = excluded.validation_json,
            metadata_json = excluded.metadata_json;
        """

        parameters = (
            evidence_uid,
            query_id,
            report_id,
            source_type,
            source_uid,
            page_number,
            table_id,
            chunk_id,
            text_content,
            lexical_score,
            vector_score,
            reranker_score,
            fusion_score,
            final_score,
            rank_position,
            int(is_valid),
            self._to_json(
                validation or {}
            ),
            self._to_json(
                metadata or {}
            ),
            now,
        )

        return self._execute_upsert_and_get_id(
            sql=sql,
            parameters=parameters,
            table_name="evidence",
            uid_column="evidence_uid",
            uid_value=evidence_uid,
            record_name="Kanıt",
        )

    def get_evidence(
        self,
        query_id: int,
        *,
        only_valid: bool = True,
    ) -> list[dict[str, Any]]:
        """Bir sorguya ait kanıtları skor sırasıyla getirir."""

        sql = """
        SELECT *
        FROM evidence
        WHERE query_id = ?
        """

        parameters: list[Any] = [
            query_id
        ]

        if only_valid:
            sql += """
            AND is_valid = 1
            """

        sql += """
        ORDER BY
            CASE
                WHEN rank_position IS NULL THEN 1
                ELSE 0
            END,
            rank_position ASC,
            final_score DESC
        """

        rows = self.fetch_all(
            sql,
            parameters,
        )

        return [
            self._deserialize_record(row)
            for row in rows
        ]

    def upsert_answer(
        self,
        *,
        answer_uid: str,
        query_id: int,
        report_id: int,
        answer_status: str,
        answer_text: str | None = None,
        confidence: float | None = None,
        model_name: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        evidence_uids: Sequence[str] | None = None,
        citations: Sequence[Any] | None = None,
        raw_response: Mapping[str, Any] | Sequence[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Nihai cevap kaydı oluşturur veya günceller."""

        now = self._utc_now()

        sql = """
        INSERT INTO answers (
            answer_uid,
            query_id,
            report_id,
            answer_text,
            answer_status,
            confidence,
            model_name,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            evidence_uids_json,
            citations_json,
            raw_response_json,
            metadata_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(answer_uid)
        DO UPDATE SET
            query_id = excluded.query_id,
            report_id = excluded.report_id,
            answer_text = excluded.answer_text,
            answer_status = excluded.answer_status,
            confidence = excluded.confidence,
            model_name = excluded.model_name,
            prompt_tokens = excluded.prompt_tokens,
            completion_tokens = excluded.completion_tokens,
            total_tokens = excluded.total_tokens,
            evidence_uids_json = excluded.evidence_uids_json,
            citations_json = excluded.citations_json,
            raw_response_json = excluded.raw_response_json,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at;
        """

        parameters = (
            answer_uid,
            query_id,
            report_id,
            answer_text,
            answer_status,
            confidence,
            model_name,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            self._to_json(
                evidence_uids or []
            ),
            self._to_json(
                citations or []
            ),
            self._to_json(
                raw_response or {}
            ),
            self._to_json(
                metadata or {}
            ),
            now,
            now,
        )

        return self._execute_upsert_and_get_id(
            sql=sql,
            parameters=parameters,
            table_name="answers",
            uid_column="answer_uid",
            uid_value=answer_uid,
            record_name="Cevap",
        )

    def get_answers(
        self,
        report_id: int,
    ) -> list[dict[str, Any]]:
        """Bir rapora ait cevapları getirir."""

        rows = self.fetch_all(
            """
            SELECT
                answers.*,
                queries.question_id,
                queries.question_text
            FROM answers
            INNER JOIN queries
                ON queries.id = answers.query_id
            WHERE answers.report_id = ?
            ORDER BY answers.created_at ASC
            """,
            (report_id,),
        )

        return [
            self._deserialize_record(row)
            for row in rows
        ]

    def clear_report_content(
        self,
        report_id: int,
        *,
        confirm: bool = False,
        preserve_queries: bool = False,
    ) -> dict[str, int]:
        """
        Rapor kaydını koruyarak ingestion tarafından üretilen alt kayıtları siler.

        `--overwrite` sırasında eski page/table/chunk/visual kayıtlarının yeni
        çalıştırmayla çakışmasını önlemek için kullanılır. İşlem tek transaction
        içinde yürütülür.

        Args:
            report_id:
                SQLite reports.id değeri.
            confirm:
                Yanlışlıkla silmeyi önleyen açık onay.
            preserve_queries:
                True ise daha önce üretilmiş query/evidence/answer kayıtlarını
                korur. Varsayılan False'tur; overwrite tam yeniden üretim kabul
                edilir.
        """

        if not confirm:
            raise DatabaseError(
                "Rapor içeriğini temizlemek için confirm=True verilmelidir."
            )

        if report_id < 1:
            raise ValueError("report_id pozitif olmalıdır.")

        counts: dict[str, int] = {}

        try:
            with self.connection() as connection:
                report_row = connection.execute(
                    "SELECT id FROM reports WHERE id = ?",
                    (report_id,),
                ).fetchone()

                if report_row is None:
                    raise RecordNotFoundError(
                        f"Rapor bulunamadı: {report_id}"
                    )

                # Görsel tablolar ana Database şeması dışında oluşturuluyor
                # olabilir; mevcutlarsa önce onları temizle.
                existing_tables = {
                    str(row["name"])
                    for row in connection.execute(
                        """
                        SELECT name
                        FROM sqlite_master
                        WHERE type = 'table'
                        """
                    ).fetchall()
                }

                for table_name, foreign_key in (
                    ("visual_metrics", "report_database_id"),
                    ("visual_assets", "report_database_id"),
                ):
                    if table_name not in existing_tables:
                        counts[table_name] = 0
                        continue

                    cursor = connection.execute(
                        f"DELETE FROM {table_name} WHERE {foreign_key} = ?",
                        (report_id,),
                    )
                    counts[table_name] = max(cursor.rowcount, 0)

                if not preserve_queries:
                    # evidence ve answers sorgulara bağlıdır; açıkça silmek
                    # sayaçların doğru raporlanmasını sağlar.
                    for table_name in ("evidence", "answers", "queries"):
                        cursor = connection.execute(
                            f"DELETE FROM {table_name} WHERE report_id = ?",
                            (report_id,),
                        )
                        counts[table_name] = max(cursor.rowcount, 0)

                # page_id foreign keyleri nedeniyle önce child tablolar.
                for table_name in ("chunks", "tables", "pages"):
                    cursor = connection.execute(
                        f"DELETE FROM {table_name} WHERE report_id = ?",
                        (report_id,),
                    )
                    counts[table_name] = max(cursor.rowcount, 0)

                connection.execute(
                    """
                    UPDATE reports
                    SET
                        status = 'processing',
                        validation_message = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        self._utc_now(),
                        report_id,
                    ),
                )

        except sqlite3.Error as exc:
            raise DatabaseError(
                "Raporun eski ingestion kayıtları temizlenemedi."
            ) from exc

        logger.info(
            "Rapor içeriği overwrite için temizlendi. report_id=%s, counts=%s",
            report_id,
            counts,
        )
        return counts

    def delete_report(
        self,
        report_id: int,
        *,
        confirm: bool = False,
    ) -> None:
        """
        Raporu ve ilişkili bütün kayıtları siler.

        Foreign key cascade nedeniyle sayfa, tablo, chunk, sorgu,
        kanıt ve cevap kayıtları da silinir.
        """

        if not confirm:
            raise DatabaseError(
                "Raporu silmek için confirm=True verilmelidir."
            )

        with self.connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM reports
                WHERE id = ?
                """,
                (report_id,),
            )

            if cursor.rowcount == 0:
                raise RecordNotFoundError(
                    f"Rapor bulunamadı: {report_id}"
                )

    def count_records(
        self,
    ) -> dict[str, int]:
        """Temel tabloların kayıt sayılarını döndürür."""

        table_names = (
            "reports",
            "pages",
            "tables",
            "chunks",
            "queries",
            "evidence",
            "answers",
        )

        counts: dict[str, int] = {}

        for table_name in table_names:
            row = self.fetch_one(
                f"""
                SELECT COUNT(*) AS record_count
                FROM {table_name}
                """
            )

            counts[table_name] = (
                int(row["record_count"])
                if row is not None
                else 0
            )

        return counts

    def vacuum(self) -> None:
        """SQLite veri tabanı dosyasını optimize eder."""

        try:
            connection = self.get_connection()

            connection.commit()
            connection.execute(
                "VACUUM;"
            )

        except sqlite3.Error as exc:
            raise DatabaseError(
                "VACUUM işlemi başarısız oldu."
            ) from exc

    def _execute_upsert_and_get_id(
        self,
        *,
        sql: str,
        parameters: Sequence[Any],
        table_name: str,
        uid_column: str,
        uid_value: str,
        record_name: str,
    ) -> int:
        """Ortak upsert işlemini çalıştırıp kayıt ID'sini döndürür."""

        try:
            with self.connection() as connection:
                connection.execute(
                    sql,
                    parameters,
                )

                row = connection.execute(
                    f"""
                    SELECT id
                    FROM {table_name}
                    WHERE {uid_column} = ?
                    """,
                    (uid_value,),
                ).fetchone()

                if row is None:
                    raise DatabaseError(
                        f"{record_name} oluşturuldu ancak ID alınamadı."
                    )

                return int(row["id"])

        except sqlite3.Error as exc:
            raise DatabaseError(
                f"{record_name} kaydı oluşturulamadı: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _row_to_dict(
        row: sqlite3.Row,
    ) -> dict[str, Any]:
        return {
            key: row[key]
            for key in row.keys()
        }

    @classmethod
    def _deserialize_record(
        cls,
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        *_json sütunlarını Python nesnelerine dönüştürür.

        SQLite boolean alanlarını da bool tipine çevirir.
        """

        result: dict[str, Any] = {}

        boolean_fields = {
            "has_native_text",
            "needs_ocr",
            "is_scanned",
            "is_valid",
        }

        for key, value in record.items():
            if key.endswith("_json"):
                output_key = key[:-5]

                if value in {
                    None,
                    "",
                }:
                    result[output_key] = None
                    continue

                try:
                    result[output_key] = json.loads(
                        value
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                ):
                    result[output_key] = value

            elif key in boolean_fields:
                result[key] = bool(value)

            else:
                result[key] = value

        return result

    @classmethod
    def _to_json(
        cls,
        value: Any,
    ) -> str:
        return json.dumps(
            cls._serialize_value(value),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def _serialize_value(
        cls,
        value: Any,
    ) -> Any:
        """Python nesnelerini JSON uyumlu hâle getirir."""

        if is_dataclass(value):
            return cls._serialize_value(
                asdict(value)
            )

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, Mapping):
            return {
                str(key): cls._serialize_value(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(value, set):
            return [
                cls._serialize_value(item)
                for item in sorted(
                    value,
                    key=str,
                )
            ]

        if isinstance(value, Sequence) and not isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            return [
                cls._serialize_value(item)
                for item in value
            ]

        return value

    @staticmethod
    def _path_to_string(
        value: str | Path | None,
    ) -> str | None:
        if value is None:
            return None

        return str(
            Path(value).expanduser()
        )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(
            UTC
        ).isoformat()

    @staticmethod
    def _validate_non_empty(
        value: str,
        field_name: str,
    ) -> None:
        if not value or not value.strip():
            raise ValueError(
                f"{field_name} boş olamaz."
            )

    @staticmethod
    def _validate_year(
        year: int,
    ) -> None:
        current_year = datetime.now(
            UTC
        ).year

        if not 1900 <= year <= current_year + 1:
            raise ValueError(
                f"Geçersiz rapor yılı: {year}"
            )


def create_database(
    database_path: str | Path = "data/sustainability_analyzer.db",
) -> Database:
    """Database sınıfı için kısa oluşturma fonksiyonu."""

    return Database(
        database_path=database_path
    )


__all__ = [
    "Database",
    "DatabaseError",
    "RecordNotFoundError",
    "create_database",
]