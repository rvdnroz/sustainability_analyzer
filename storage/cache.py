from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar


logger = logging.getLogger(__name__)


T = TypeVar("T")


class CacheError(RuntimeError):
    """Önbellek işlemi başarısız olduğunda yükseltilir."""


class CacheNamespace(StrEnum):
    """Sistemde kullanılan temel önbellek alanları."""

    EXA = "exa"
    HTTP = "http"
    URL_VALIDATION = "url_validation"
    PDF_VALIDATION = "pdf_validation"
    OCR = "ocr"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    GEMINI = "gemini"
    TABLE_EXTRACTION = "table_extraction"
    PAGE_CLASSIFICATION = "page_classification"
    LAYOUT = "layout"
    RETRIEVAL = "retrieval"
    QUERY_EXPANSION = "query_expansion"
    OTHER = "other"


@dataclass(slots=True, frozen=True)
class CacheEntry:
    """Önbellekte bulunan tek bir kaydın metadata bilgileri."""

    namespace: CacheNamespace
    key: str
    hashed_key: str

    data_path: Path
    metadata_path: Path

    created_at: datetime
    expires_at: datetime | None
    last_accessed_at: datetime | None

    hit_count: int
    size_bytes: int

    is_expired: bool

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace.value,
            "key": self.key,
            "hashed_key": self.hashed_key,
            "data_path": str(self.data_path),
            "metadata_path": str(self.metadata_path),
            "created_at": self.created_at.isoformat(),
            "expires_at": (
                self.expires_at.isoformat()
                if self.expires_at is not None
                else None
            ),
            "last_accessed_at": (
                self.last_accessed_at.isoformat()
                if self.last_accessed_at is not None
                else None
            ),
            "hit_count": self.hit_count,
            "size_bytes": self.size_bytes,
            "is_expired": self.is_expired,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class CacheStats:
    """Önbelleğin genel durum özeti."""

    entry_count: int
    expired_entry_count: int
    total_size_bytes: int

    namespaces: dict[str, int]
    namespace_sizes: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "expired_entry_count": self.expired_entry_count,
            "total_size_bytes": self.total_size_bytes,
            "namespaces": self.namespaces,
            "namespace_sizes": self.namespace_sizes,
        }


class FileCache:
    """
    Dosya tabanlı, namespace destekli önbellek sistemi.

    Örnek klasör yapısı:

    data/
    └── cache/
        ├── exa/
        │   ├── ab/
        │   │   ├── abcdef....json
        │   │   └── abcdef....metadata.json
        ├── ocr/
        ├── embedding/
        ├── gemini/
        └── retrieval/

    Temel özellikler:

    - SHA-256 tabanlı deterministik cache anahtarları
    - Namespace ayrımı
    - TTL desteği
    - Atomik dosya yazımı
    - JSON uyumlu veri saklama
    - Cache hit sayacı
    - Süresi dolmuş kayıtları temizleme
    - Namespace bazlı temizleme
    - Aynı işlem tekrarlandığında hesaplamayı atlama
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        *,
        default_ttl_seconds: int | None = None,
        update_access_metadata: bool = True,
        create_directory: bool = True,
    ) -> None:
        if (
            default_ttl_seconds is not None
            and default_ttl_seconds <= 0
        ):
            raise ValueError(
                "default_ttl_seconds pozitif olmalıdır."
            )

        self.cache_dir = Path(
            cache_dir
        ).expanduser().resolve()

        self.default_ttl_seconds = default_ttl_seconds
        self.update_access_metadata = update_access_metadata

        self._lock = threading.RLock()

        if create_directory:
            self.cache_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

    def get(
        self,
        namespace: CacheNamespace | str,
        key: Any,
        *,
        default: T | None = None,
        allow_expired: bool = False,
    ) -> Any | T | None:
        """
        Önbellek kaydını döndürür.

        Kayıt bulunamazsa veya süresi dolmuşsa default döndürülür.
        """

        normalized_namespace = self._normalize_namespace(
            namespace
        )

        normalized_key = self.build_key(
            key
        )

        hashed_key = self.hash_key(
            normalized_key
        )

        data_path, metadata_path = self._entry_paths(
            normalized_namespace,
            hashed_key,
        )

        if not data_path.is_file():
            return default

        if not metadata_path.is_file():
            logger.warning(
                "Cache metadata dosyası eksik: %s",
                metadata_path,
            )

            return default

        with self._lock:
            metadata = self._read_metadata(
                metadata_path
            )

            if metadata is None:
                return default

            is_expired = self._is_metadata_expired(
                metadata
            )

            if is_expired and not allow_expired:
                self.delete(
                    normalized_namespace,
                    normalized_key,
                    missing_ok=True,
                )

                return default

            try:
                with data_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    payload = json.load(file)

            except json.JSONDecodeError as exc:
                raise CacheError(
                    f"Cache verisi geçersiz JSON: {data_path}"
                ) from exc

            except OSError as exc:
                raise CacheError(
                    f"Cache verisi okunamadı: {data_path}"
                ) from exc

            if self.update_access_metadata:
                metadata["last_accessed_at"] = (
                    self._utc_now().isoformat()
                )

                metadata["hit_count"] = (
                    int(
                        metadata.get(
                            "hit_count",
                            0,
                        )
                    )
                    + 1
                )

                self._write_json_atomic(
                    metadata_path,
                    metadata,
                    overwrite=True,
                )

            return payload.get(
                "value",
                default,
            )

    def set(
        self,
        namespace: CacheNamespace | str,
        key: Any,
        value: Any,
        *,
        ttl_seconds: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = True,
    ) -> CacheEntry:
        """
        Bir değeri önbelleğe kaydeder.

        ttl_seconds None ise sınıf seviyesindeki default TTL kullanılır.
        Hem parametre hem default TTL None ise kayıt süresizdir.
        """

        normalized_namespace = self._normalize_namespace(
            namespace
        )

        normalized_key = self.build_key(
            key
        )

        hashed_key = self.hash_key(
            normalized_key
        )

        data_path, metadata_path = self._entry_paths(
            normalized_namespace,
            hashed_key,
        )

        effective_ttl = (
            ttl_seconds
            if ttl_seconds is not None
            else self.default_ttl_seconds
        )

        if (
            effective_ttl is not None
            and effective_ttl <= 0
        ):
            raise ValueError(
                "ttl_seconds pozitif olmalıdır."
            )

        created_at = self._utc_now()

        expires_at = (
            created_at
            + timedelta(
                seconds=effective_ttl
            )
            if effective_ttl is not None
            else None
        )

        payload = {
            "value": self._serialize_value(
                value
            )
        }

        metadata_payload = {
            "namespace": normalized_namespace.value,
            "key": normalized_key,
            "hashed_key": hashed_key,
            "created_at": created_at.isoformat(),
            "expires_at": (
                expires_at.isoformat()
                if expires_at is not None
                else None
            ),
            "last_accessed_at": None,
            "hit_count": 0,
            "ttl_seconds": effective_ttl,
            "metadata": self._serialize_value(
                metadata or {}
            ),
        }

        with self._lock:
            self._write_json_atomic(
                data_path,
                payload,
                overwrite=overwrite,
            )

            metadata_payload["size_bytes"] = (
                data_path.stat().st_size
            )

            self._write_json_atomic(
                metadata_path,
                metadata_payload,
                overwrite=overwrite,
            )

        return self._build_cache_entry(
            namespace=normalized_namespace,
            key=normalized_key,
            hashed_key=hashed_key,
            data_path=data_path,
            metadata_path=metadata_path,
            metadata_payload=metadata_payload,
        )

    def get_or_set(
        self,
        namespace: CacheNamespace | str,
        key: Any,
        factory: Callable[[], T],
        *,
        ttl_seconds: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        cache_none: bool = False,
    ) -> T:
        """
        Kayıt varsa önbellekten döndürür, yoksa factory fonksiyonunu çalıştırır.

        Bu metot pahalı işlemlerin tekrarını önlemek için kullanılır.
        """

        marker = object()

        cached_value = self.get(
            namespace,
            key,
            default=marker,
        )

        if cached_value is not marker:
            return cached_value

        value = factory()

        if value is not None or cache_none:
            self.set(
                namespace=namespace,
                key=key,
                value=value,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            )

        return value

    def contains(
        self,
        namespace: CacheNamespace | str,
        key: Any,
        *,
        include_expired: bool = False,
    ) -> bool:
        """Cache kaydının mevcut olup olmadığını kontrol eder."""

        normalized_namespace = self._normalize_namespace(
            namespace
        )

        normalized_key = self.build_key(
            key
        )

        hashed_key = self.hash_key(
            normalized_key
        )

        data_path, metadata_path = self._entry_paths(
            normalized_namespace,
            hashed_key,
        )

        if not data_path.is_file():
            return False

        if not metadata_path.is_file():
            return False

        metadata = self._read_metadata(
            metadata_path
        )

        if metadata is None:
            return False

        if (
            self._is_metadata_expired(metadata)
            and not include_expired
        ):
            return False

        return True

    def get_entry(
        self,
        namespace: CacheNamespace | str,
        key: Any,
    ) -> CacheEntry | None:
        """Cache değerini okumadan metadata kaydını döndürür."""

        normalized_namespace = self._normalize_namespace(
            namespace
        )

        normalized_key = self.build_key(
            key
        )

        hashed_key = self.hash_key(
            normalized_key
        )

        data_path, metadata_path = self._entry_paths(
            normalized_namespace,
            hashed_key,
        )

        if (
            not data_path.is_file()
            or not metadata_path.is_file()
        ):
            return None

        metadata_payload = self._read_metadata(
            metadata_path
        )

        if metadata_payload is None:
            return None

        return self._build_cache_entry(
            namespace=normalized_namespace,
            key=normalized_key,
            hashed_key=hashed_key,
            data_path=data_path,
            metadata_path=metadata_path,
            metadata_payload=metadata_payload,
        )

    def delete(
        self,
        namespace: CacheNamespace | str,
        key: Any,
        *,
        missing_ok: bool = True,
    ) -> None:
        """Tek bir cache kaydını siler."""

        normalized_namespace = self._normalize_namespace(
            namespace
        )

        normalized_key = self.build_key(
            key
        )

        hashed_key = self.hash_key(
            normalized_key
        )

        data_path, metadata_path = self._entry_paths(
            normalized_namespace,
            hashed_key,
        )

        existing_files = [
            path
            for path in (
                data_path,
                metadata_path,
            )
            if path.exists()
        ]

        if not existing_files:
            if missing_ok:
                return

            raise FileNotFoundError(
                "Silinecek cache kaydı bulunamadı: "
                f"{normalized_namespace.value}/{hashed_key}"
            )

        with self._lock:
            for path in existing_files:
                try:
                    path.unlink(
                        missing_ok=True
                    )

                except OSError as exc:
                    raise CacheError(
                        f"Cache dosyası silinemedi: {path}"
                    ) from exc

    def clear_namespace(
        self,
        namespace: CacheNamespace | str,
        *,
        confirm: bool = False,
    ) -> int:
        """Belirtilen namespace içindeki bütün kayıtları siler."""

        if not confirm:
            raise CacheError(
                "Namespace temizlemek için confirm=True verilmelidir."
            )

        normalized_namespace = self._normalize_namespace(
            namespace
        )

        namespace_dir = (
            self.cache_dir
            / normalized_namespace.value
        )

        if not namespace_dir.exists():
            return 0

        file_count = sum(
            path.is_file()
            for path in namespace_dir.rglob("*")
        )

        try:
            shutil.rmtree(
                namespace_dir
            )

        except OSError as exc:
            raise CacheError(
                f"Namespace temizlenemedi: {namespace_dir}"
            ) from exc

        return file_count

    def clear_all(
        self,
        *,
        confirm: bool = False,
    ) -> int:
        """Bütün cache kayıtlarını siler."""

        if not confirm:
            raise CacheError(
                "Tüm cache'i temizlemek için confirm=True verilmelidir."
            )

        if not self.cache_dir.exists():
            return 0

        file_count = sum(
            path.is_file()
            for path in self.cache_dir.rglob("*")
        )

        try:
            shutil.rmtree(
                self.cache_dir
            )

            self.cache_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        except OSError as exc:
            raise CacheError(
                f"Cache klasörü temizlenemedi: {self.cache_dir}"
            ) from exc

        return file_count

    def cleanup_expired(self) -> int:
        """Süresi dolmuş bütün cache kayıtlarını siler."""

        removed_count = 0

        for metadata_path in self.cache_dir.rglob(
            "*.metadata.json"
        ):
            metadata = self._read_metadata(
                metadata_path
            )

            if metadata is None:
                continue

            if not self._is_metadata_expired(
                metadata
            ):
                continue

            data_path = metadata_path.with_name(
                metadata_path.name.replace(
                    ".metadata.json",
                    ".json",
                )
            )

            try:
                data_path.unlink(
                    missing_ok=True
                )

                metadata_path.unlink(
                    missing_ok=True
                )

                removed_count += 1

            except OSError as exc:
                logger.warning(
                    "Süresi dolmuş cache kaydı silinemedi: %s",
                    exc,
                )

        return removed_count

    def cleanup_older_than(
        self,
        age_seconds: int,
    ) -> int:
        """Belirtilen yaştan daha eski cache kayıtlarını siler."""

        if age_seconds <= 0:
            raise ValueError(
                "age_seconds pozitif olmalıdır."
            )

        threshold = (
            self._utc_now()
            - timedelta(
                seconds=age_seconds
            )
        )

        removed_count = 0

        for metadata_path in self.cache_dir.rglob(
            "*.metadata.json"
        ):
            metadata = self._read_metadata(
                metadata_path
            )

            if metadata is None:
                continue

            created_at = self._parse_datetime(
                metadata.get("created_at")
            )

            if created_at is None:
                continue

            if created_at >= threshold:
                continue

            data_path = metadata_path.with_name(
                metadata_path.name.replace(
                    ".metadata.json",
                    ".json",
                )
            )

            try:
                data_path.unlink(
                    missing_ok=True
                )

                metadata_path.unlink(
                    missing_ok=True
                )

                removed_count += 1

            except OSError as exc:
                logger.warning(
                    "Eski cache kaydı silinemedi: %s",
                    exc,
                )

        return removed_count

    def list_entries(
        self,
        namespace: CacheNamespace | str | None = None,
        *,
        include_expired: bool = True,
    ) -> list[CacheEntry]:
        """Cache kayıtlarını metadata bilgileriyle listeler."""

        if namespace is None:
            search_root = self.cache_dir
        else:
            normalized_namespace = self._normalize_namespace(
                namespace
            )

            search_root = (
                self.cache_dir
                / normalized_namespace.value
            )

        if not search_root.exists():
            return []

        entries: list[CacheEntry] = []

        for metadata_path in search_root.rglob(
            "*.metadata.json"
        ):
            metadata = self._read_metadata(
                metadata_path
            )

            if metadata is None:
                continue

            namespace_value = metadata.get(
                "namespace"
            )

            try:
                entry_namespace = CacheNamespace(
                    namespace_value
                )
            except ValueError:
                entry_namespace = CacheNamespace.OTHER

            hashed_key = str(
                metadata.get(
                    "hashed_key",
                    "",
                )
            )

            data_path = metadata_path.with_name(
                metadata_path.name.replace(
                    ".metadata.json",
                    ".json",
                )
            )

            if not data_path.is_file():
                continue

            entry = self._build_cache_entry(
                namespace=entry_namespace,
                key=str(
                    metadata.get(
                        "key",
                        "",
                    )
                ),
                hashed_key=hashed_key,
                data_path=data_path,
                metadata_path=metadata_path,
                metadata_payload=metadata,
            )

            if (
                entry.is_expired
                and not include_expired
            ):
                continue

            entries.append(entry)

        return sorted(
            entries,
            key=lambda entry: entry.created_at,
            reverse=True,
        )

    def stats(self) -> CacheStats:
        """Cache kayıtlarının genel istatistiklerini döndürür."""

        entries = self.list_entries(
            include_expired=True
        )

        namespace_counts: dict[str, int] = {}
        namespace_sizes: dict[str, int] = {}

        for entry in entries:
            namespace_name = (
                entry.namespace.value
            )

            namespace_counts[
                namespace_name
            ] = (
                namespace_counts.get(
                    namespace_name,
                    0,
                )
                + 1
            )

            namespace_sizes[
                namespace_name
            ] = (
                namespace_sizes.get(
                    namespace_name,
                    0,
                )
                + entry.size_bytes
            )

        return CacheStats(
            entry_count=len(entries),
            expired_entry_count=sum(
                entry.is_expired
                for entry in entries
            ),
            total_size_bytes=sum(
                entry.size_bytes
                for entry in entries
            ),
            namespaces=namespace_counts,
            namespace_sizes=namespace_sizes,
        )

    @classmethod
    def build_key(
        cls,
        value: Any,
    ) -> str:
        """
        Her tür Python girdisinden deterministik bir cache anahtarı üretir.

        Dictionary anahtarları sıralanır. Path, Enum, dataclass ve datetime
        değerleri JSON uyumlu biçime dönüştürülür.
        """

        serialized = cls._serialize_value(
            value
        )

        try:
            return json.dumps(
                serialized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        except TypeError as exc:
            raise CacheError(
                "Cache anahtarı JSON biçimine dönüştürülemedi."
            ) from exc

    @staticmethod
    def hash_key(
        key: str,
    ) -> str:
        """Cache anahtarının SHA-256 özetini üretir."""

        return hashlib.sha256(
            key.encode("utf-8")
        ).hexdigest()

    def _entry_paths(
        self,
        namespace: CacheNamespace,
        hashed_key: str,
    ) -> tuple[Path, Path]:
        """
        Cache kaydı için veri ve metadata yollarını üretir.

        İlk iki hash karakteri alt klasör olarak kullanılır.
        """

        prefix = hashed_key[:2]

        entry_dir = (
            self.cache_dir
            / namespace.value
            / prefix
        )

        entry_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        data_path = (
            entry_dir
            / f"{hashed_key}.json"
        )

        metadata_path = (
            entry_dir
            / f"{hashed_key}.metadata.json"
        )

        return (
            data_path,
            metadata_path,
        )

    def _build_cache_entry(
        self,
        *,
        namespace: CacheNamespace,
        key: str,
        hashed_key: str,
        data_path: Path,
        metadata_path: Path,
        metadata_payload: Mapping[str, Any],
    ) -> CacheEntry:
        created_at = self._parse_datetime(
            metadata_payload.get(
                "created_at"
            )
        )

        if created_at is None:
            created_at = datetime.fromtimestamp(
                data_path.stat().st_ctime,
                tz=UTC,
            )

        expires_at = self._parse_datetime(
            metadata_payload.get(
                "expires_at"
            )
        )

        last_accessed_at = self._parse_datetime(
            metadata_payload.get(
                "last_accessed_at"
            )
        )

        return CacheEntry(
            namespace=namespace,
            key=key,
            hashed_key=hashed_key,
            data_path=data_path,
            metadata_path=metadata_path,
            created_at=created_at,
            expires_at=expires_at,
            last_accessed_at=last_accessed_at,
            hit_count=int(
                metadata_payload.get(
                    "hit_count",
                    0,
                )
            ),
            size_bytes=int(
                metadata_payload.get(
                    "size_bytes",
                    (
                        data_path.stat().st_size
                        if data_path.exists()
                        else 0
                    ),
                )
            ),
            is_expired=(
                expires_at is not None
                and self._utc_now()
                >= expires_at
            ),
            metadata=dict(
                metadata_payload.get(
                    "metadata",
                    {},
                )
            ),
        )

    def _read_metadata(
        self,
        metadata_path: Path,
    ) -> dict[str, Any] | None:
        try:
            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except FileNotFoundError:
            return None

        except json.JSONDecodeError as exc:
            logger.warning(
                "Geçersiz cache metadata dosyası: %s (%s)",
                metadata_path,
                exc,
            )

            return None

        except OSError as exc:
            raise CacheError(
                f"Cache metadata okunamadı: {metadata_path}"
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            logger.warning(
                "Cache metadata nesnesi sözlük değil: %s",
                metadata_path,
            )

            return None

        return data

    def _is_metadata_expired(
        self,
        metadata: Mapping[str, Any],
    ) -> bool:
        expires_at = self._parse_datetime(
            metadata.get(
                "expires_at"
            )
        )

        if expires_at is None:
            return False

        return self._utc_now() >= expires_at

    def _write_json_atomic(
        self,
        target_path: Path,
        data: Any,
        *,
        overwrite: bool,
    ) -> None:
        if (
            target_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"Cache dosyası zaten mevcut: {target_path}"
            )

        target_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target_path.parent,
                prefix=f".{target_path.name}.",
                suffix=".part",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(
                    temporary_file.name
                )

                json.dump(
                    data,
                    temporary_file,
                    ensure_ascii=False,
                    indent=2,
                )

                temporary_file.flush()
                os.fsync(
                    temporary_file.fileno()
                )

            os.replace(
                temporary_path,
                target_path,
            )

        except OSError as exc:
            raise CacheError(
                f"Cache dosyası yazılamadı: {target_path}"
            ) from exc

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink(
                    missing_ok=True
                )

    @staticmethod
    def _normalize_namespace(
        namespace: CacheNamespace | str,
    ) -> CacheNamespace:
        if isinstance(
            namespace,
            CacheNamespace,
        ):
            return namespace

        try:
            return CacheNamespace(
                namespace
            )

        except ValueError:
            logger.warning(
                "Bilinmeyen cache namespace '%s', OTHER kullanılıyor.",
                namespace,
            )

            return CacheNamespace.OTHER

    @classmethod
    def _serialize_value(
        cls,
        value: Any,
    ) -> Any:
        """Python nesnelerini JSON uyumlu biçime dönüştürür."""

        if is_dataclass(value):
            return cls._serialize_value(
                asdict(value)
            )

        if isinstance(value, Path):
            return str(
                value.expanduser()
            )

        if isinstance(value, datetime):
            return value.astimezone(
                UTC
            ).isoformat()

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, Mapping):
            return {
                str(key): cls._serialize_value(
                    item
                )
                for key, item in sorted(
                    value.items(),
                    key=lambda item: str(
                        item[0]
                    ),
                )
            }

        if isinstance(value, set):
            return [
                cls._serialize_value(
                    item
                )
                for item in sorted(
                    value,
                    key=str,
                )
            ]

        if isinstance(
            value,
            Sequence,
        ) and not isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            return [
                cls._serialize_value(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            bytes,
        ):
            return {
                "__type__": "bytes",
                "sha256": hashlib.sha256(
                    value
                ).hexdigest(),
                "size": len(value),
            }

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ) or value is None:
            return value

        return repr(value)

    @staticmethod
    def _parse_datetime(
        value: Any,
    ) -> datetime | None:
        if value in {
            None,
            "",
        }:
            return None

        if isinstance(
            value,
            datetime,
        ):
            if value.tzinfo is None:
                return value.replace(
                    tzinfo=UTC
                )

            return value.astimezone(
                UTC
            )

        try:
            parsed = datetime.fromisoformat(
                str(value)
            )

        except ValueError:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=UTC
            )

        return parsed.astimezone(
            UTC
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(
            UTC
        )


def create_cache(
    cache_dir: str | Path = "data/cache",
    *,
    default_ttl_seconds: int | None = None,
) -> FileCache:
    """FileCache sınıfı için kısa oluşturma fonksiyonu."""

    return FileCache(
        cache_dir=cache_dir,
        default_ttl_seconds=default_ttl_seconds,
    )


__all__ = [
    "CacheEntry",
    "CacheError",
    "CacheNamespace",
    "CacheStats",
    "FileCache",
    "create_cache",
]
