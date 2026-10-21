from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(slots=True, frozen=True)
class VectorDocument:
    document_id: str
    text: str
    embedding: tuple[float, ...]
    report_id: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    category: str | None = None
    standard_codes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class VectorSearchResult:
    document_id: str
    score: float
    document: VectorDocument
    rank: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class VectorRetrieverConfig:
    top_k: int = 20
    minimum_score: float = 0.0
    normalize_embeddings: bool = True
    similarity_metric: str = "cosine"

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k pozitif olmalıdır.")

        if not -1.0 <= self.minimum_score <= 1.0:
            raise ValueError(
                "minimum_score -1 ile 1 arasında olmalıdır.",
            )

        supported_metrics = {
            "cosine",
            "dot_product",
            "euclidean",
        }

        if self.similarity_metric not in supported_metrics:
            raise ValueError(
                "similarity_metric şu değerlerden biri olmalıdır: "
                f"{sorted(supported_metrics)}",
            )


class EmbeddingProviderProtocol(Protocol):
    def embed_query(self, text: str) -> Sequence[float]:
        ...

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        ...


class VectorStoreProtocol(Protocol):
    def add(
        self,
        document: Any,
        *,
        overwrite: bool = False,
    ) -> Any:
        ...

    def add_many(
        self,
        documents: Iterable[Any],
        *,
        overwrite: bool = False,
    ) -> int:
        ...

    def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        report_id: str | None = None,
        category: str | None = None,
        standard_codes: Sequence[str] = (),
    ) -> Iterable[Any]:
        ...

    def get_optional(
        self,
        document_id: str,
    ) -> Any | None:
        ...

    def all(self) -> Iterable[Any]:
        ...

    def clear(self) -> None:
        ...


class VectorRetrieverError(RuntimeError):
    """Vektör retrieval işlemleri sırasında oluşan temel hata."""


class EmbeddingDimensionError(VectorRetrieverError):
    """Embedding boyutları uyumsuz olduğunda yükseltilir."""


class EmbeddingProviderNotConfiguredError(VectorRetrieverError):
    """Embedding sağlayıcısı tanımlanmadan embedding istendiğinde yükseltilir."""


class VectorRetriever:
    """
    Embedding tabanlı anlamsal retrieval işlemlerini yönetir.

    Retriever şu özellikleri destekler:
    - Chunk veya belge nesnelerinden embedding indeksi oluşturma
    - Sorgu metnini embedding vektörüne dönüştürme
    - Cosine similarity, dot product ve Euclidean benzerliği
    - Rapor, sayfa, kategori ve standart kodu filtreleri
    - Harici vector store veya bellek içi indeks kullanımı
    - Önceden hesaplanmış embeddinglerle çalışma
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProviderProtocol | None = None,
        *,
        vector_store: VectorStoreProtocol | None = None,
        config: VectorRetrieverConfig | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.config = config or VectorRetrieverConfig()

        self._documents: dict[str, VectorDocument] = {}
        self._embedding_dimension: int | None = None

    def build_index(
        self,
        documents: Iterable[Any],
        *,
        overwrite: bool = False,
        batch_size: int = 32,
    ) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size pozitif olmalıdır.")

        source_documents = list(documents)

        if not source_documents:
            return 0

        prepared_documents: list[VectorDocument] = []
        pending_documents: list[tuple[str, str, dict[str, Any]]] = []

        for source in source_documents:
            document_data = self._extract_document_data(source)

            existing_embedding = document_data.pop("embedding", None)

            if existing_embedding is not None:
                prepared_documents.append(
                    self._create_vector_document(
                        embedding=existing_embedding,
                        **document_data,
                    ),
                )
            else:
                pending_documents.append(
                    (
                        document_data["document_id"],
                        document_data["text"],
                        document_data,
                    ),
                )

        if pending_documents:
            self._ensure_embedding_provider()

            for start_index in range(
                0,
                len(pending_documents),
                batch_size,
            ):
                batch = pending_documents[
                    start_index : start_index + batch_size
                ]

                batch_texts = [
                    item[1]
                    for item in batch
                ]

                embeddings = self.embedding_provider.embed_documents(
                    batch_texts,
                )

                if len(embeddings) != len(batch):
                    raise VectorRetrieverError(
                        "Embedding sağlayıcısının döndürdüğü vektör sayısı "
                        "belge sayısıyla eşleşmiyor.",
                    )

                for (_, _, document_data), embedding in zip(
                    batch,
                    embeddings,
                    strict=True,
                ):
                    prepared_documents.append(
                        self._create_vector_document(
                            embedding=embedding,
                            **document_data,
                        ),
                    )

        if self.vector_store is not None:
            add_many = getattr(self.vector_store, "add_many", None)
            if callable(add_many):
                return int(
                    add_many(
                        prepared_documents,
                        overwrite=overwrite,
                    )
                )

            add = getattr(self.vector_store, "add", None)
            if not callable(add):
                raise VectorRetrieverError(
                    "Vector store üzerinde add veya add_many metodu "
                    "bulunamadı.",
                )

            added_count = 0
            for prepared_document in prepared_documents:
                add(
                    prepared_document,
                    overwrite=overwrite,
                )
                added_count += 1
            return added_count

        added_count = 0

        for document in prepared_documents:
            if (
                document.document_id in self._documents
                and not overwrite
            ):
                raise VectorRetrieverError(
                    f"'{document.document_id}' kimliğine sahip belge "
                    "zaten indekslenmiş.",
                )

            self._documents[document.document_id] = document
            added_count += 1

        return added_count

    def add_document(
        self,
        document: Any,
        *,
        embedding: Sequence[float] | None = None,
        overwrite: bool = False,
    ) -> VectorDocument:
        document_data = self._extract_document_data(document)

        resolved_embedding = (
            embedding
            if embedding is not None
            else document_data.pop("embedding", None)
        )

        if resolved_embedding is None:
            self._ensure_embedding_provider()
            resolved_embedding = self.embedding_provider.embed_query(
                document_data["text"],
            )

        vector_document = self._create_vector_document(
            embedding=resolved_embedding,
            **document_data,
        )

        if self.vector_store is not None:
            add = getattr(self.vector_store, "add", None)
            if not callable(add):
                raise VectorRetrieverError(
                    "Vector store üzerinde add metodu bulunamadı.",
                )

            add(
                vector_document,
                overwrite=overwrite,
            )
            return vector_document

        if (
            vector_document.document_id in self._documents
            and not overwrite
        ):
            raise VectorRetrieverError(
                f"'{vector_document.document_id}' kimliğine sahip belge "
                "zaten indekslenmiş.",
            )

        self._documents[vector_document.document_id] = vector_document

        return vector_document

    def retrieve(
        self,
        query: str | Sequence[float],
        *,
        top_k: int | None = None,
        minimum_score: float | None = None,
        report_id: str | None = None,
        page_numbers: Sequence[int] = (),
        category: str | None = None,
        standard_codes: Sequence[str] = (),
    ) -> list[VectorSearchResult]:
        query_embedding = self._resolve_query_embedding(query)

        result_limit = top_k or self.config.top_k
        score_threshold = (
            self.config.minimum_score
            if minimum_score is None
            else minimum_score
        )

        if result_limit <= 0:
            return []

        if not -1.0 <= score_threshold <= 1.0:
            raise ValueError(
                "minimum_score -1 ile 1 arasında olmalıdır.",
            )

        if self.vector_store is not None:
            return self._search_vector_store(
                query_embedding,
                top_k=result_limit,
                minimum_score=score_threshold,
                report_id=report_id,
                page_numbers=page_numbers,
                category=category,
                standard_codes=standard_codes,
            )

        results: list[VectorSearchResult] = []

        for document in self._documents.values():
            if not self._matches_filters(
                document,
                report_id=report_id,
                page_numbers=page_numbers,
                category=category,
                standard_codes=standard_codes,
            ):
                continue

            score = self._calculate_similarity(
                query_embedding,
                document.embedding,
            )

            if score < score_threshold:
                continue

            results.append(
                VectorSearchResult(
                    document_id=document.document_id,
                    score=round(score, 6),
                    document=document,
                    metadata={
                        "similarity_metric": (
                            self.config.similarity_metric
                        ),
                        "embedding_dimension": len(
                            document.embedding,
                        ),
                    },
                ),
            )

        results.sort(
            key=lambda result: (
                -result.score,
                result.document.page_number
                if result.document.page_number is not None
                else 10**9,
                result.document_id,
            ),
        )

        return [
            VectorSearchResult(
                document_id=result.document_id,
                score=result.score,
                document=result.document,
                rank=index + 1,
                metadata=result.metadata,
            )
            for index, result in enumerate(
                results[:result_limit],
            )
        ]

    def search(
        self,
        query: str | Sequence[float],
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        return self.retrieve(query, **kwargs)

    def retrieve_by_embedding(
        self,
        embedding: Sequence[float],
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        return self.retrieve(embedding, **kwargs)

    def get_document(
        self,
        document_id: str,
    ) -> VectorDocument:
        normalized_id = self._normalize_identifier(document_id)

        if self.vector_store is not None:
            for source in self.vector_store.all():
                document = self._to_vector_document(source)

                if document.document_id == normalized_id:
                    return document

            raise VectorRetrieverError(
                f"'{normalized_id}' kimliğine sahip belge bulunamadı.",
            )

        try:
            return self._documents[normalized_id]
        except KeyError as exc:
            raise VectorRetrieverError(
                f"'{normalized_id}' kimliğine sahip belge bulunamadı.",
            ) from exc

    def all_documents(self) -> list[VectorDocument]:
        if self.vector_store is not None:
            documents = [
                self._to_vector_document(source)
                for source in self.vector_store.all()
            ]
        else:
            documents = list(self._documents.values())

        return sorted(
            documents,
            key=lambda document: (
                document.report_id or "",
                document.page_number
                if document.page_number is not None
                else 0,
                document.document_id,
            ),
        )

    def remove_document(
        self,
        document_id: str,
    ) -> VectorDocument:
        if self.vector_store is not None:
            delete_method = getattr(
                self.vector_store,
                "delete",
                None,
            )

            if not callable(delete_method):
                raise VectorRetrieverError(
                    "Vector store delete metodu sağlamıyor.",
                )

            deleted = delete_method(document_id)

            return self._to_vector_document(deleted)

        normalized_id = self._normalize_identifier(document_id)

        try:
            return self._documents.pop(normalized_id)
        except KeyError as exc:
            raise VectorRetrieverError(
                f"'{normalized_id}' kimliğine sahip belge bulunamadı.",
            ) from exc

    def count(self) -> int:
        if self.vector_store is not None:
            count_method = getattr(
                self.vector_store,
                "count",
                None,
            )

            if callable(count_method):
                return int(count_method())

            return len(list(self.vector_store.all()))

        return len(self._documents)

    def clear(self) -> None:
        if self.vector_store is not None:
            self.vector_store.clear()
        else:
            self._documents.clear()

        self._embedding_dimension = None

    def embed_query(
        self,
        query: str,
    ) -> tuple[float, ...]:
        self._ensure_embedding_provider()

        cleaned_query = self._clean_text(query)

        if not cleaned_query:
            raise VectorRetrieverError(
                "Embedding üretilecek sorgu boş olamaz.",
            )

        embedding = self.embedding_provider.embed_query(
            cleaned_query,
        )

        return self._prepare_embedding(embedding)

    def _search_vector_store(
        self,
        query_embedding: tuple[float, ...],
        *,
        top_k: int,
        minimum_score: float,
        report_id: str | None,
        page_numbers: Sequence[int],
        category: str | None,
        standard_codes: Sequence[str],
    ) -> list[VectorSearchResult]:
        store_results = self.vector_store.search(
            query_embedding,
            top_k=max(top_k * 3, top_k),
            report_id=report_id,
            category=category,
            standard_codes=standard_codes,
        )

        results: list[VectorSearchResult] = []

        for source in store_results:
            document = self._extract_store_document(source)
            score = self._extract_store_score(
                source,
                query_embedding=query_embedding,
                document=document,
            )

            if not self._matches_filters(
                document,
                report_id=report_id,
                page_numbers=page_numbers,
                category=category,
                standard_codes=standard_codes,
            ):
                continue

            if score < minimum_score:
                continue

            results.append(
                VectorSearchResult(
                    document_id=document.document_id,
                    score=round(score, 6),
                    document=document,
                    metadata={
                        "similarity_metric": (
                            self.config.similarity_metric
                        ),
                        "source": "vector_store",
                    },
                ),
            )

        results.sort(
            key=lambda result: (
                -result.score,
                result.document.page_number
                if result.document.page_number is not None
                else 10**9,
                result.document_id,
            ),
        )

        return [
            VectorSearchResult(
                document_id=result.document_id,
                score=result.score,
                document=result.document,
                rank=index + 1,
                metadata=result.metadata,
            )
            for index, result in enumerate(
                results[:top_k],
            )
        ]

    def _calculate_similarity(
        self,
        query_embedding: Sequence[float],
        document_embedding: Sequence[float],
    ) -> float:
        self._validate_embedding_dimensions(
            query_embedding,
            document_embedding,
        )

        if self.config.similarity_metric == "cosine":
            return self._cosine_similarity(
                query_embedding,
                document_embedding,
            )

        if self.config.similarity_metric == "dot_product":
            return self._dot_product(
                query_embedding,
                document_embedding,
            )

        if self.config.similarity_metric == "euclidean":
            distance = self._euclidean_distance(
                query_embedding,
                document_embedding,
            )
            return 1.0 / (1.0 + distance)

        raise VectorRetrieverError(
            "Desteklenmeyen benzerlik metriği: "
            f"{self.config.similarity_metric}",
        )

    def _resolve_query_embedding(
        self,
        query: str | Sequence[float],
    ) -> tuple[float, ...]:
        if isinstance(query, str):
            return self.embed_query(query)

        return self._prepare_embedding(query)

    def _create_vector_document(
        self,
        *,
        document_id: str,
        text: str,
        embedding: Sequence[float],
        report_id: str | None = None,
        page_number: int | None = None,
        section_title: str | None = None,
        category: str | None = None,
        standard_codes: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> VectorDocument:
        prepared_embedding = self._prepare_embedding(
            embedding,
        )

        return VectorDocument(
            document_id=self._normalize_identifier(document_id),
            text=self._clean_text(text),
            embedding=prepared_embedding,
            report_id=self._optional_string(report_id),
            page_number=self._optional_int(page_number),
            section_title=self._optional_string(section_title),
            category=self._optional_string(category),
            standard_codes=self._normalize_string_sequence(
                standard_codes,
            ),
            metadata=dict(metadata or {}),
        )

    def _extract_document_data(
        self,
        source: Any,
    ) -> dict[str, Any]:
        if isinstance(source, VectorDocument):
            return {
                "document_id": source.document_id,
                "text": source.text,
                "embedding": source.embedding,
                "report_id": source.report_id,
                "page_number": source.page_number,
                "section_title": source.section_title,
                "category": source.category,
                "standard_codes": source.standard_codes,
                "metadata": dict(source.metadata),
            }

        metadata = self._get_metadata(source)

        document_id = self._get_first_value(
            source,
            (
                "chunk_id",
                "document_id",
                "id",
                "identifier",
            ),
        )

        if document_id is None:
            document_id = (
                metadata.get("chunk_id")
                or metadata.get("document_id")
                or metadata.get("id")
            )

        text = self._get_first_value(
            source,
            (
                "text",
                "content",
                "chunk_text",
            ),
        )

        if text is None:
            text = (
                metadata.get("text")
                or metadata.get("content")
            )

        if document_id is None or not str(document_id).strip():
            raise VectorRetrieverError(
                "Belge üzerinde chunk_id, document_id veya id bulunamadı.",
            )

        cleaned_text = self._clean_text(text)

        if not cleaned_text:
            raise VectorRetrieverError(
                f"'{document_id}' kimliğine sahip belgenin metni boş.",
            )

        embedding = self._get_first_value(
            source,
            (
                "embedding",
                "vector",
                "embedding_vector",
            ),
        )

        if embedding is None:
            embedding = (
                metadata.get("embedding")
                or metadata.get("vector")
            )

        report_id = self._get_first_value(
            source,
            ("report_id",),
        )
        report_id = report_id or metadata.get("report_id")

        page_number = self._get_first_value(
            source,
            (
                "page_number",
                "source_page",
                "page",
            ),
        )
        page_number = (
            page_number
            if page_number is not None
            else (
                metadata.get("page_number")
                or metadata.get("source_page")
                or metadata.get("page")
            )
        )

        section_title = self._get_first_value(
            source,
            (
                "section_title",
                "section",
                "heading",
            ),
        )
        section_title = (
            section_title
            or metadata.get("section_title")
            or metadata.get("section")
        )

        category = self._get_first_value(
            source,
            (
                "category",
                "esg_category",
            ),
        )
        category = (
            category
            or metadata.get("category")
            or metadata.get("esg_category")
        )

        standard_codes = self._get_first_value(
            source,
            (
                "standard_codes",
                "standards",
                "standard_references",
            ),
        )
        standard_codes = (
            standard_codes
            if standard_codes is not None
            else (
                metadata.get("standard_codes")
                or metadata.get("standards")
                or ()
            )
        )

        return {
            "document_id": self._normalize_identifier(document_id),
            "text": cleaned_text,
            "embedding": embedding,
            "report_id": self._optional_string(report_id),
            "page_number": self._optional_int(page_number),
            "section_title": self._optional_string(section_title),
            "category": self._optional_string(category),
            "standard_codes": self._normalize_string_sequence(
                standard_codes,
            ),
            "metadata": metadata,
        }

    def _to_vector_document(
        self,
        source: Any,
    ) -> VectorDocument:
        if isinstance(source, VectorDocument):
            return source

        data = self._extract_document_data(source)
        embedding = data.pop("embedding", None)

        if embedding is None:
            raise VectorRetrieverError(
                "Vector store kaydında embedding bulunamadı.",
            )

        return self._create_vector_document(
            embedding=embedding,
            **data,
        )

    def _extract_store_document(
        self,
        source: Any,
    ) -> VectorDocument:
        if isinstance(source, VectorSearchResult):
            return source.document

        document = self._get_first_value(
            source,
            (
                "document",
                "item",
                "record",
            ),
        )

        if document is not None:
            return self._to_vector_document(document)

        # storage.vector_store.VectorSearchResult yalnızca kimlik, skor,
        # metin ve metadata taşır; embedding vektörünü taşımaz. Retriever'ın
        # kendi VectorDocument modelini oluşturabilmesi için tam kaydı
        # vector store'dan document_id ile tekrar alırız.
        source_embedding = self._get_first_value(
            source,
            (
                "embedding",
                "vector",
                "embedding_vector",
            ),
        )

        if source_embedding is None and self.vector_store is not None:
            document_id = self._get_first_value(
                source,
                (
                    "document_id",
                    "chunk_id",
                    "table_id",
                    "id",
                ),
            )

            if document_id is not None:
                stored_document = None

                get_optional = getattr(
                    self.vector_store,
                    "get_optional",
                    None,
                )
                if callable(get_optional):
                    stored_document = get_optional(str(document_id))

                if stored_document is None:
                    get = getattr(self.vector_store, "get", None)
                    if callable(get):
                        try:
                            stored_document = get(str(document_id))
                        except Exception:
                            stored_document = None

                if stored_document is not None:
                    return self._to_vector_document(stored_document)

        return self._to_vector_document(source)

    def _extract_store_score(
        self,
        source: Any,
        *,
        query_embedding: Sequence[float],
        document: VectorDocument,
    ) -> float:
        if isinstance(source, VectorSearchResult):
            return float(source.score)

        score = self._get_first_value(
            source,
            (
                "score",
                "similarity",
                "distance",
            ),
        )

        if score is None:
            return self._calculate_similarity(
                query_embedding,
                document.embedding,
            )

        numeric_score = float(score)

        if (
            self.config.similarity_metric == "euclidean"
            and self._get_first_value(source, ("distance",)) is not None
        ):
            return 1.0 / (1.0 + max(numeric_score, 0.0))

        return numeric_score

    def _matches_filters(
        self,
        document: VectorDocument,
        *,
        report_id: str | None,
        page_numbers: Sequence[int],
        category: str | None,
        standard_codes: Sequence[str],
    ) -> bool:
        if (
            report_id is not None
            and document.report_id != report_id
        ):
            return False

        if (
            page_numbers
            and document.page_number not in set(page_numbers)
        ):
            return False

        if (
            category is not None
            and self._normalize_text(document.category or "")
            != self._normalize_text(category)
        ):
            return False

        if standard_codes:
            document_codes = {
                self._normalize_standard_code(code)
                for code in document.standard_codes
            }
            required_codes = {
                self._normalize_standard_code(code)
                for code in standard_codes
            }

            if not required_codes.intersection(document_codes):
                return False

        return True

    def _prepare_embedding(
        self,
        embedding: Sequence[float],
    ) -> tuple[float, ...]:
        try:
            vector = tuple(float(value) for value in embedding)
        except (TypeError, ValueError) as exc:
            raise VectorRetrieverError(
                "Embedding sayısal değerlerden oluşmalıdır.",
            ) from exc

        if not vector:
            raise VectorRetrieverError(
                "Embedding vektörü boş olamaz.",
            )

        if any(
            math.isnan(value) or math.isinf(value)
            for value in vector
        ):
            raise VectorRetrieverError(
                "Embedding NaN veya sonsuz değer içeremez.",
            )

        if self._embedding_dimension is None:
            self._embedding_dimension = len(vector)
        elif len(vector) != self._embedding_dimension:
            raise EmbeddingDimensionError(
                "Embedding boyutu indeks boyutuyla uyuşmuyor. "
                f"Beklenen: {self._embedding_dimension}, "
                f"gelen: {len(vector)}",
            )

        if self.config.normalize_embeddings:
            vector = self._normalize_vector(vector)

        return vector

    @staticmethod
    def _normalize_vector(
        vector: Sequence[float],
    ) -> tuple[float, ...]:
        norm = math.sqrt(
            sum(value * value for value in vector),
        )

        if norm == 0.0:
            raise VectorRetrieverError(
                "Sıfır normlu embedding kullanılamaz.",
            )

        return tuple(
            value / norm
            for value in vector
        )

    @staticmethod
    def _validate_embedding_dimensions(
        first: Sequence[float],
        second: Sequence[float],
    ) -> None:
        if len(first) != len(second):
            raise EmbeddingDimensionError(
                "Karşılaştırılan embedding boyutları farklı. "
                f"{len(first)} != {len(second)}",
            )

    @staticmethod
    def _cosine_similarity(
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        dot_product = sum(
            left * right
            for left, right in zip(
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

        denominator = first_norm * second_norm

        if denominator == 0.0:
            return 0.0

        return max(
            -1.0,
            min(dot_product / denominator, 1.0),
        )

    @staticmethod
    def _dot_product(
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        return sum(
            left * right
            for left, right in zip(
                first,
                second,
                strict=True,
            )
        )

    @staticmethod
    def _euclidean_distance(
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        return math.sqrt(
            sum(
                (left - right) ** 2
                for left, right in zip(
                    first,
                    second,
                    strict=True,
                )
            ),
        )

    def _ensure_embedding_provider(self) -> None:
        if self.embedding_provider is None:
            raise EmbeddingProviderNotConfiguredError(
                "Embedding sağlayıcısı tanımlanmamış.",
            )

    @staticmethod
    def _get_metadata(source: Any) -> dict[str, Any]:
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
    def _get_first_value(
        source: Any,
        names: Sequence[str],
    ) -> Any:
        for name in names:
            if isinstance(source, Mapping):
                value = source.get(name)
            else:
                value = getattr(source, name, None)

            if value is not None:
                return getattr(value, "value", value)

        return None

    @staticmethod
    def _normalize_identifier(value: Any) -> str:
        if value is None or not str(value).strip():
            raise ValueError("Belge kimliği boş olamaz.")

        return str(value).strip()

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None

        enum_value = getattr(value, "value", value)
        cleaned = str(enum_value).strip()

        return cleaned or None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _normalize_string_sequence(
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

        return tuple(
            sorted(
                {
                    cls._clean_text(item)
                    for item in iterable
                    if cls._clean_text(item)
                },
            ),
        )

    @staticmethod
    def _normalize_standard_code(value: str) -> str:
        return " ".join(
            str(value).strip().upper().split(),
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

        return " ".join(
            str(value)
            .translate(translation_table)
            .casefold()
            .split(),
        )


__all__ = [
    "EmbeddingDimensionError",
    "EmbeddingProviderNotConfiguredError",
    "EmbeddingProviderProtocol",
    "VectorDocument",
    "VectorRetriever",
    "VectorRetrieverConfig",
    "VectorRetrieverError",
    "VectorSearchResult",
    "VectorStoreProtocol",
]