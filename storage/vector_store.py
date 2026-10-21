from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(slots=True, frozen=True)
class VectorDocument:
    document_id: str
    embedding: tuple[float, ...]
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError(
                "document_id boş olamaz.",
            )

        if not self.embedding:
            raise ValueError(
                "embedding boş olamaz.",
            )

        if not all(
            math.isfinite(float(value))
            for value in self.embedding
        ):
            raise ValueError(
                "embedding yalnızca sonlu sayılar içermelidir.",
            )


@dataclass(slots=True, frozen=True)
class VectorSearchResult:
    document_id: str
    score: float
    rank: int
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class VectorStoreStats:
    document_count: int
    embedding_dimension: int | None
    collection_name: str
    persistent: bool
    storage_path: str | None = None


@dataclass(slots=True, frozen=True)
class VectorStoreConfig:
    collection_name: str = "sustainability_documents"
    persist_directory: str | Path | None = None

    similarity_metric: str = "cosine"
    normalize_embeddings: bool = True
    auto_persist: bool = True

    def __post_init__(self) -> None:
        if not self.collection_name.strip():
            raise ValueError(
                "collection_name boş olamaz.",
            )

        normalized_metric = self.similarity_metric.strip().casefold()

        if normalized_metric not in {
            "cosine",
            "dot_product",
            "euclidean",
        }:
            raise ValueError(
                "similarity_metric yalnızca cosine, dot_product "
                "veya euclidean olabilir.",
            )


class EmbeddingProviderProtocol(Protocol):
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        ...

    def embed_query(
        self,
        text: str,
    ) -> Sequence[float]:
        ...


class HashingEmbeddingProvider:
    """Harici model gerektirmeyen deterministik yerel embedding sağlayıcısı.

    Bu sağlayıcı üretim kalitesinde bir dil modeli yerine geçmez; geliştirme,
    entegrasyon testi ve ücretsiz yerel çalışma için sözcük/karakter n-gram
    hashing yaklaşımı kullanır. Aynı metin her zaman aynı vektörü üretir.
    """

    _TOKEN_RE = re.compile(r"[\wÀ-ž]+", re.UNICODE)

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 32:
            raise ValueError("Embedding boyutu en az 32 olmalıdır.")
        self.dimension = int(dimension)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> Sequence[float]:
        cleaned = " ".join(str(text or "").casefold().split())
        vector = [0.0] * self.dimension
        if not cleaned:
            return vector

        tokens = self._TOKEN_RE.findall(cleaned)
        features = list(tokens)
        features.extend(
            f"{tokens[index]}::{tokens[index + 1]}"
            for index in range(max(len(tokens) - 1, 0))
        )

        for feature in features:
            digest = hashlib.blake2b(
                feature.encode("utf-8"),
                digest_size=16,
            ).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector


class VectorStoreError(RuntimeError):
    """Vector store işlemlerinde oluşan temel hata."""


class VectorDimensionError(VectorStoreError):
    """Embedding boyutları eşleşmediğinde yükseltilir."""


class VectorDocumentNotFoundError(VectorStoreError):
    """İstenen vector document bulunamadığında yükseltilir."""


class VectorStore:
    """
    Chunk, tablo satırı ve standart kayıtlarının embedding vektörlerini
    saklayan bağımlılıksız vector store uygulaması.

    Bu sınıfın sorumlulukları:
    - Embedding ve metadata kayıtlarını saklamak
    - Cosine, dot product veya Euclidean benzerliği ile arama yapmak
    - Metadata filtreleri uygulamak
    - Kayıtları JSON dosyasına kalıcı olarak yazmak
    - VectorRetriever için ortak arama arayüzü sağlamak

    Üretim ortamında ChromaDB veya pgvector kullanılmadan önce yerel
    geliştirme ve test amacıyla kullanılabilir.
    """

    _STORE_FILENAME = "vector_store.json"

    def __init__(
        self,
        *,
        config: VectorStoreConfig | None = None,
        embedding_provider: EmbeddingProviderProtocol | None = None,
    ) -> None:
        self.config = config or VectorStoreConfig()
        self.embedding_provider = embedding_provider

        self._documents: dict[str, VectorDocument] = {}
        self._embedding_dimension: int | None = None
        self._lock = threading.RLock()

        self._persist_directory = self._resolve_persist_directory(
            self.config.persist_directory,
        )

        if self._persist_directory is not None:
            self._persist_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.load()

    def add(
        self,
        document: VectorDocument,
        *,
        overwrite: bool = False,
    ) -> None:
        prepared_document = self._prepare_document(document)

        with self._lock:
            if (
                not overwrite
                and prepared_document.document_id
                in self._documents
            ):
                raise VectorStoreError(
                    f"'{prepared_document.document_id}' kimlikli kayıt "
                    "zaten mevcut.",
                )

            self._validate_dimension(
                prepared_document.embedding,
            )

            self._documents[
                prepared_document.document_id
            ] = prepared_document

            if self.config.auto_persist:
                self.persist()

    def add_many(
        self,
        documents: Iterable[VectorDocument],
        *,
        overwrite: bool = False,
    ) -> int:
        prepared_documents = [
            self._prepare_document(document)
            for document in documents
        ]

        if not prepared_documents:
            return 0

        dimensions = {
            len(document.embedding)
            for document in prepared_documents
        }

        if len(dimensions) != 1:
            raise VectorDimensionError(
                "Eklenecek embedding vektörlerinin boyutları eşleşmiyor.",
            )

        target_dimension = next(iter(dimensions))

        with self._lock:
            if (
                self._embedding_dimension is not None
                and target_dimension != self._embedding_dimension
            ):
                raise VectorDimensionError(
                    "Yeni embedding boyutu mevcut store boyutuyla "
                    "eşleşmiyor.",
                )

            if not overwrite:
                duplicate_ids = [
                    document.document_id
                    for document in prepared_documents
                    if document.document_id in self._documents
                ]

                if duplicate_ids:
                    raise VectorStoreError(
                        "Aşağıdaki kayıt kimlikleri zaten mevcut: "
                        f"{', '.join(sorted(duplicate_ids))}",
                    )

            for document in prepared_documents:
                self._documents[
                    document.document_id
                ] = document

            self._embedding_dimension = target_dimension

            if self.config.auto_persist:
                self.persist()

        return len(prepared_documents)

    def add_text(
        self,
        document_id: str,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> VectorDocument:
        if self.embedding_provider is None:
            raise VectorStoreError(
                "Metinden embedding üretmek için embedding_provider "
                "tanımlanmalıdır.",
            )

        cleaned_text = self._clean_text(text)

        if not cleaned_text:
            raise ValueError(
                "Embedding üretilecek metin boş olamaz.",
            )

        embedding = tuple(
            float(value)
            for value in self.embedding_provider.embed_query(
                cleaned_text,
            )
        )

        if self._is_zero_norm(embedding):
            raise VectorStoreError(
                "Embedding sıfır normlu olduğu için kayıt indekslenmedi.",
            )

        document = VectorDocument(
            document_id=document_id,
            embedding=embedding,
            text=cleaned_text,
            metadata=dict(metadata or {}),
        )

        self.add(
            document,
            overwrite=overwrite,
        )

        return document

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        document_ids: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
        overwrite: bool = False,
    ) -> int:
        if self.embedding_provider is None:
            raise VectorStoreError(
                "Metinlerden embedding üretmek için embedding_provider "
                "tanımlanmalıdır.",
            )

        if len(texts) != len(document_ids):
            raise ValueError(
                "texts ve document_ids uzunlukları eşit olmalıdır.",
            )

        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(
                "metadatas ve texts uzunlukları eşit olmalıdır.",
            )

        prepared_rows: list[
            tuple[str, str, Mapping[str, Any]]
        ] = []

        for index, text in enumerate(texts):
            cleaned_text = self._clean_text(text)

            if not cleaned_text:
                continue

            prepared_rows.append(
                (
                    str(document_ids[index]),
                    cleaned_text,
                    (
                        metadatas[index]
                        if metadatas is not None
                        else {}
                    ),
                )
            )

        if not prepared_rows:
            return 0

        embeddings = self.embedding_provider.embed_documents(
            [row[1] for row in prepared_rows],
        )

        if len(embeddings) != len(prepared_rows):
            raise VectorStoreError(
                "Embedding sağlayıcısı beklenen sayıda vektör döndürmedi.",
            )

        documents: list[VectorDocument] = []

        for index, embedding_values in enumerate(embeddings):
            embedding = tuple(
                float(value)
                for value in embedding_values
            )

            if self._is_zero_norm(embedding):
                continue

            document_id, cleaned_text, metadata = prepared_rows[index]

            documents.append(
                VectorDocument(
                    document_id=document_id,
                    embedding=embedding,
                    text=cleaned_text,
                    metadata=dict(metadata),
                )
            )

        if not documents:
            return 0

        return self.add_many(
            documents,
            overwrite=overwrite,
        )

    def get(
        self,
        document_id: str,
    ) -> VectorDocument:
        cleaned_id = self._clean_text(document_id)

        with self._lock:
            document = self._documents.get(cleaned_id)

        if document is None:
            raise VectorDocumentNotFoundError(
                f"'{cleaned_id}' kimlikli vector kayıt bulunamadı.",
            )

        return document

    def get_optional(
        self,
        document_id: str,
    ) -> VectorDocument | None:
        cleaned_id = self._clean_text(document_id)

        with self._lock:
            return self._documents.get(cleaned_id)

    def contains(
        self,
        document_id: str,
    ) -> bool:
        cleaned_id = self._clean_text(document_id)

        with self._lock:
            return cleaned_id in self._documents

    def delete(
        self,
        document_id: str,
        *,
        missing_ok: bool = False,
    ) -> bool:
        cleaned_id = self._clean_text(document_id)

        with self._lock:
            if cleaned_id not in self._documents:
                if missing_ok:
                    return False

                raise VectorDocumentNotFoundError(
                    f"'{cleaned_id}' kimlikli vector kayıt bulunamadı.",
                )

            del self._documents[cleaned_id]

            if not self._documents:
                self._embedding_dimension = None

            if self.config.auto_persist:
                self.persist()

        return True

    def delete_many(
        self,
        document_ids: Iterable[str],
        *,
        missing_ok: bool = True,
    ) -> int:
        deleted_count = 0

        with self._lock:
            for document_id in document_ids:
                cleaned_id = self._clean_text(document_id)

                if cleaned_id not in self._documents:
                    if not missing_ok:
                        raise VectorDocumentNotFoundError(
                            f"'{cleaned_id}' kimlikli kayıt bulunamadı.",
                        )

                    continue

                del self._documents[cleaned_id]
                deleted_count += 1

            if not self._documents:
                self._embedding_dimension = None

            if deleted_count and self.config.auto_persist:
                self.persist()

        return deleted_count

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._embedding_dimension = None

            if self.config.auto_persist:
                self.persist()

    def all(
        self,
        *,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> tuple[VectorDocument, ...]:
        with self._lock:
            documents = tuple(self._documents.values())

        if metadata_filter is None:
            return documents

        return tuple(
            document
            for document in documents
            if self._matches_metadata_filter(
                document.metadata,
                metadata_filter,
            )
        )

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 10,
        metadata_filter: Mapping[str, Any] | None = None,
        minimum_score: float | None = None,
        exclude_document_ids: Iterable[str] = (),
        report_id: str | None = None,
        category: str | None = None,
        standard_codes: Sequence[str] = (),
    ) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError(
                "top_k pozitif olmalıdır.",
            )

        resolved_filter = dict(metadata_filter or {})
        if report_id is not None:
            resolved_filter["report_id"] = str(report_id)
        if category is not None:
            resolved_filter["category"] = str(category)

        prepared_query = self._prepare_embedding(
            query_embedding,
        )

        self._validate_dimension(
            prepared_query,
            set_dimension=False,
        )

        excluded_ids = {
            self._clean_text(document_id)
            for document_id in exclude_document_ids
            if self._clean_text(document_id)
        }

        with self._lock:
            documents = tuple(self._documents.values())

        scored_documents: list[
            tuple[VectorDocument, float]
        ] = []

        for document in documents:
            if document.document_id in excluded_ids:
                continue

            if resolved_filter and not self._matches_metadata_filter(
                document.metadata,
                resolved_filter,
            ):
                continue

            if standard_codes:
                document_codes = {
                    str(value).casefold()
                    for value in document.metadata.get("standards", ())
                }
                requested_codes = {
                    str(value).casefold()
                    for value in standard_codes
                }
                if requested_codes and not requested_codes.intersection(document_codes):
                    continue

            score = self._calculate_similarity(
                prepared_query,
                document.embedding,
            )

            if (
                minimum_score is not None
                and score < minimum_score
            ):
                continue

            scored_documents.append(
                (
                    document,
                    score,
                ),
            )

        scored_documents.sort(
            key=lambda item: (
                -item[1],
                item[0].document_id,
            ),
        )

        return [
            VectorSearchResult(
                document_id=document.document_id,
                score=round(score, 8),
                rank=index,
                text=document.text,
                metadata=document.metadata,
            )
            for index, (document, score) in enumerate(
                scored_documents[:top_k],
                start=1,
            )
        ]

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 10,
        metadata_filter: Mapping[str, Any] | None = None,
        minimum_score: float | None = None,
        exclude_document_ids: Iterable[str] = (),
    ) -> list[VectorSearchResult]:
        if self.embedding_provider is None:
            raise VectorStoreError(
                "Metin sorgusu için embedding_provider tanımlanmalıdır.",
            )

        cleaned_query = self._clean_text(query)

        if not cleaned_query:
            raise ValueError(
                "Vector arama sorgusu boş olamaz.",
            )

        query_embedding = self.embedding_provider.embed_query(
            cleaned_query,
        )

        return self.search(
            query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            minimum_score=minimum_score,
            exclude_document_ids=exclude_document_ids,
        )

    def count(
        self,
        *,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> int:
        if metadata_filter is None:
            with self._lock:
                return len(self._documents)

        return len(
            self.all(
                metadata_filter=metadata_filter,
            ),
        )

    def stats(self) -> VectorStoreStats:
        with self._lock:
            document_count = len(self._documents)
            embedding_dimension = self._embedding_dimension

        return VectorStoreStats(
            document_count=document_count,
            embedding_dimension=embedding_dimension,
            collection_name=self.config.collection_name,
            persistent=self._persist_directory is not None,
            storage_path=(
                str(self._store_file_path)
                if self._persist_directory is not None
                else None
            ),
        )

    def persist(self) -> None:
        if self._persist_directory is None:
            return

        with self._lock:
            payload = {
                "version": 1,
                "collection_name": self.config.collection_name,
                "similarity_metric": self.config.similarity_metric,
                "normalize_embeddings": (
                    self.config.normalize_embeddings
                ),
                "embedding_dimension": self._embedding_dimension,
                "documents": [
                    {
                        "document_id": document.document_id,
                        "embedding": list(document.embedding),
                        "text": document.text,
                        "metadata": self._make_json_serializable(
                            document.metadata,
                        ),
                    }
                    for document in self._documents.values()
                ],
            }

            target_path = self._store_file_path
            temporary_path = target_path.with_name(
                f"{target_path.name}."
                f"{os.getpid()}."
                f"{threading.get_ident()}."
                f"{uuid.uuid4().hex}.tmp"
            )

            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )

            try:
                with temporary_path.open(
                    "w",
                    encoding="utf-8",
                    newline="\n",
                ) as file:
                    file.write(serialized)
                    file.flush()
                    os.fsync(file.fileno())

                last_error: OSError | None = None

                for attempt in range(6):
                    try:
                        os.replace(
                            temporary_path,
                            target_path,
                        )
                        last_error = None
                        break
                    except OSError as exc:
                        last_error = exc
                        time.sleep(
                            0.05 * (2 ** attempt)
                        )

                if last_error is not None:
                    raise VectorStoreError(
                        "Vector store dosyası atomik olarak "
                        f"güncellenemedi: {last_error}"
                    ) from last_error

            finally:
                try:
                    if temporary_path.exists():
                        temporary_path.unlink()
                except OSError:
                    pass

    def load(self) -> int:
        if self._persist_directory is None:
            return 0

        if not self._store_file_path.exists():
            return 0

        try:
            payload = json.loads(
                self._store_file_path.read_text(
                    encoding="utf-8",
                ),
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(
                "Vector store dosyası okunamadı.",
            ) from exc

        documents_data = payload.get("documents", [])

        if not isinstance(documents_data, list):
            raise VectorStoreError(
                "Vector store dosyasındaki documents alanı geçersiz.",
            )

        loaded_documents: dict[str, VectorDocument] = {}
        detected_dimension: int | None = None

        for item in documents_data:
            if not isinstance(item, Mapping):
                continue

            try:
                document = VectorDocument(
                    document_id=str(
                        item["document_id"],
                    ),
                    embedding=tuple(
                        float(value)
                        for value in item["embedding"]
                    ),
                    text=self._clean_text(
                        item.get("text"),
                    ),
                    metadata=dict(
                        item.get("metadata") or {},
                    ),
                )

                document = self._prepare_document(document)
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise VectorStoreError(
                    "Vector store dosyasında geçersiz kayıt bulundu.",
                ) from exc

            if detected_dimension is None:
                detected_dimension = len(
                    document.embedding,
                )
            elif len(document.embedding) != detected_dimension:
                raise VectorDimensionError(
                    "Kalıcı vector store içindeki embedding boyutları "
                    "eşleşmiyor.",
                )

            loaded_documents[
                document.document_id
            ] = document

        with self._lock:
            self._documents = loaded_documents
            self._embedding_dimension = detected_dimension

        return len(loaded_documents)

    def _prepare_document(
        self,
        document: VectorDocument,
    ) -> VectorDocument:
        prepared_embedding = self._prepare_embedding(
            document.embedding,
        )

        return VectorDocument(
            document_id=self._clean_text(
                document.document_id,
            ),
            embedding=prepared_embedding,
            text=self._clean_text(
                document.text,
            ),
            metadata=dict(document.metadata),
        )

    def _prepare_embedding(
        self,
        embedding: Sequence[float],
    ) -> tuple[float, ...]:
        if not embedding:
            raise ValueError(
                "Embedding boş olamaz.",
            )

        prepared = tuple(
            float(value)
            for value in embedding
        )

        if not all(
            math.isfinite(value)
            for value in prepared
        ):
            raise ValueError(
                "Embedding yalnızca sonlu sayılar içermelidir.",
            )

        if (
            self.config.normalize_embeddings
            and self.config.similarity_metric.casefold()
            in {
                "cosine",
                "dot_product",
            }
        ):
            return self._normalize_vector(prepared)

        return prepared

    def _validate_dimension(
        self,
        embedding: Sequence[float],
        *,
        set_dimension: bool = True,
    ) -> None:
        dimension = len(embedding)

        if dimension <= 0:
            raise VectorDimensionError(
                "Embedding boyutu sıfır olamaz.",
            )

        if self._embedding_dimension is None:
            if set_dimension:
                self._embedding_dimension = dimension

            return

        if dimension != self._embedding_dimension:
            raise VectorDimensionError(
                "Embedding boyutu vector store boyutuyla eşleşmiyor. "
                f"Beklenen: {self._embedding_dimension}, "
                f"gelen: {dimension}.",
            )

    def _calculate_similarity(
        self,
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        metric = self.config.similarity_metric.casefold()

        if metric == "cosine":
            return self._cosine_similarity(
                first,
                second,
            )

        if metric == "dot_product":
            return sum(
                first_value * second_value
                for first_value, second_value in zip(
                    first,
                    second,
                    strict=True,
                )
            )

        if metric == "euclidean":
            distance = math.sqrt(
                sum(
                    (
                        first_value
                        - second_value
                    )
                    ** 2
                    for first_value, second_value in zip(
                        first,
                        second,
                        strict=True,
                    )
                ),
            )

            return 1.0 / (1.0 + distance)

        raise VectorStoreError(
            f"Desteklenmeyen similarity metriği: {metric}",
        )

    @staticmethod
    def _cosine_similarity(
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        dot_product = sum(
            first_value * second_value
            for first_value, second_value in zip(
                first,
                second,
                strict=True,
            )
        )

        first_norm = math.sqrt(
            sum(value * value for value in first),
        )

        second_norm = math.sqrt(
            sum(value * value for value in second),
        )

        if first_norm == 0.0 or second_norm == 0.0:
            return 0.0

        return dot_product / (
            first_norm * second_norm
        )

    @staticmethod
    def _is_zero_norm(
        embedding: Sequence[float],
        *,
        epsilon: float = 1e-12,
    ) -> bool:
        return (
            math.sqrt(
                sum(
                    float(value) * float(value)
                    for value in embedding
                )
            )
            <= epsilon
        )

    @staticmethod
    def _normalize_vector(
        embedding: Sequence[float],
    ) -> tuple[float, ...]:
        norm = math.sqrt(
            sum(value * value for value in embedding),
        )

        if norm == 0.0:
            raise ValueError(
                "Sıfır normlu embedding normalize edilemez.",
            )

        return tuple(
            value / norm
            for value in embedding
        )

    @classmethod
    def _matches_metadata_filter(
        cls,
        metadata: Mapping[str, Any],
        metadata_filter: Mapping[str, Any],
    ) -> bool:
        for key, expected_value in metadata_filter.items():
            actual_value = cls._get_nested_value(
                metadata,
                key,
            )

            if callable(expected_value):
                try:
                    if not bool(expected_value(actual_value)):
                        return False
                except Exception:
                    return False

                continue

            if isinstance(
                expected_value,
                (set, list, tuple),
            ):
                if isinstance(
                    actual_value,
                    (set, list, tuple),
                ):
                    if not set(expected_value).intersection(
                        actual_value,
                    ):
                        return False
                elif actual_value not in expected_value:
                    return False

                continue

            if actual_value != expected_value:
                return False

        return True

    @staticmethod
    def _get_nested_value(
        mapping: Mapping[str, Any],
        dotted_key: str,
    ) -> Any:
        current: Any = mapping

        for part in dotted_key.split("."):
            if not isinstance(current, Mapping):
                return None

            current = current.get(part)

        return current

    @staticmethod
    def _resolve_persist_directory(
        path: str | Path | None,
    ) -> Path | None:
        if path is None:
            return None

        return Path(path).expanduser().resolve()

    @property
    def _store_file_path(self) -> Path:
        if self._persist_directory is None:
            raise VectorStoreError(
                "Kalıcı vector store dizini tanımlanmamış.",
            )

        collection_name = (
            self.config.collection_name
            .replace("/", "_")
            .replace("\\", "_")
            .strip()
        )

        return self._persist_directory / (
            f"{collection_name}_{self._STORE_FILENAME}"
        )

    @classmethod
    def _make_json_serializable(
        cls,
        value: Any,
    ) -> Any:
        if value is None or isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, Mapping):
            return {
                str(key): cls._make_json_serializable(
                    item,
                )
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                cls._make_json_serializable(item)
                for item in value
            ]

        enum_value = getattr(value, "value", None)

        if enum_value is not None:
            return cls._make_json_serializable(
                enum_value,
            )

        if hasattr(value, "model_dump"):
            return cls._make_json_serializable(
                value.model_dump(
                    exclude_none=True,
                ),
            )

        if hasattr(value, "dict"):
            return cls._make_json_serializable(
                value.dict(
                    exclude_none=True,
                ),
            )

        return str(value)

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:
        if value is None:
            return ""

        return " ".join(
            str(value).replace("\n", " ").split(),
        )


__all__ = [
    "EmbeddingProviderProtocol",
    "VectorDimensionError",
    "VectorDocument",
    "VectorDocumentNotFoundError",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreConfig",
    "VectorStoreError",
    "VectorStoreStats",
]