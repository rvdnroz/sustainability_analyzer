from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(slots=True, frozen=True)
class BM25Document:
    document_id: str
    text: str
    report_id: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    category: str | None = None
    standard_codes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BM25SearchResult:
    document_id: str
    score: float
    document: BM25Document
    matched_terms: tuple[str, ...] = ()
    rank: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BM25RetrieverConfig:
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 20
    minimum_score: float = 0.0
    lowercase: bool = True
    remove_stopwords: bool = True
    minimum_token_length: int = 2

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("k1 pozitif olmalıdır.")

        if not 0.0 <= self.b <= 1.0:
            raise ValueError("b değeri 0 ile 1 arasında olmalıdır.")

        if self.top_k <= 0:
            raise ValueError("top_k pozitif olmalıdır.")

        if self.minimum_score < 0:
            raise ValueError("minimum_score negatif olamaz.")

        if self.minimum_token_length <= 0:
            raise ValueError(
                "minimum_token_length pozitif olmalıdır.",
            )


class ChunkRepositoryProtocol(Protocol):
    def all(self) -> Iterable[Any]:
        ...

    def find(
        self,
        *,
        filters: Any | None = None,
        text_query: str | None = None,
        limit: int | None = None,
    ) -> Iterable[Any]:
        ...


class BM25RetrieverError(RuntimeError):
    """BM25 retrieval işlemleri sırasında oluşan temel hata."""


class BM25IndexNotBuiltError(BM25RetrieverError):
    """BM25 indeksi oluşturulmadan arama yapıldığında yükseltilir."""


class BM25Retriever:
    """
    Chunk ve metin belgeleri üzerinde BM25 tabanlı sözcüksel arama yapar.

    Retriever şu özellikleri destekler:
    - Türkçe ve İngilizce metin tokenizasyonu
    - BM25 Okapi skorlaması
    - Rapor, sayfa, kategori ve standart kodu filtreleri
    - Sorgu terimi ağırlıklandırma
    - Dinamik belge ekleme ve indeks yenileme
    - Chunk modellerini ortak BM25Document yapısına dönüştürme
    """

    _TOKEN_PATTERN = re.compile(
        r"[a-zA-ZçÇğĞıİöÖşŞüÜ0-9]+"
        r"(?:[-./][a-zA-ZçÇğĞıİöÖşŞüÜ0-9]+)*",
    )

    _TURKISH_STOPWORDS = {
        "acaba",
        "ama",
        "ancak",
        "artık",
        "aslında",
        "az",
        "bazı",
        "belki",
        "ben",
        "bile",
        "bir",
        "biraz",
        "biz",
        "bu",
        "buna",
        "bunda",
        "bundan",
        "bunlar",
        "çok",
        "çünkü",
        "da",
        "daha",
        "de",
        "değil",
        "diğer",
        "diye",
        "en",
        "gibi",
        "hem",
        "hep",
        "hepsi",
        "her",
        "hiç",
        "için",
        "ile",
        "ise",
        "ki",
        "kim",
        "mı",
        "mi",
        "mu",
        "mü",
        "nasıl",
        "ne",
        "neden",
        "nerede",
        "olan",
        "olarak",
        "oldu",
        "olduğu",
        "olmak",
        "olur",
        "o",
        "onlar",
        "sonra",
        "şey",
        "şu",
        "ve",
        "veya",
        "ya",
        "yani",
    }

    _ENGLISH_STOPWORDS = {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "more",
        "most",
        "not",
        "of",
        "on",
        "or",
        "other",
        "our",
        "out",
        "over",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "under",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    }

    def __init__(
        self,
        repository: ChunkRepositoryProtocol | None = None,
        *,
        config: BM25RetrieverConfig | None = None,
    ) -> None:
        self.repository = repository
        self.config = config or BM25RetrieverConfig()

        self._documents: dict[str, BM25Document] = {}
        self._document_tokens: dict[str, tuple[str, ...]] = {}
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._document_frequencies: Counter[str] = Counter()

        self._document_lengths: dict[str, int] = {}
        self._average_document_length: float = 0.0
        self._is_built = False

    def build_index(
        self,
        documents: Iterable[BM25Document | Any] | None = None,
    ) -> int:
        source_documents = (
            list(documents)
            if documents is not None
            else list(self._load_repository_documents())
        )

        self.clear()

        for document in source_documents:
            normalized_document = self._to_document(document)
            self._documents[normalized_document.document_id] = (
                normalized_document
            )

        self._rebuild_statistics()

        return len(self._documents)

    def add_document(
        self,
        document: BM25Document | Any,
        *,
        overwrite: bool = False,
        rebuild: bool = True,
    ) -> BM25Document:
        normalized_document = self._to_document(document)

        if (
            normalized_document.document_id in self._documents
            and not overwrite
        ):
            raise BM25RetrieverError(
                f"'{normalized_document.document_id}' kimliğine sahip "
                "belge zaten indekslenmiş.",
            )

        self._documents[normalized_document.document_id] = (
            normalized_document
        )

        if rebuild:
            self._rebuild_statistics()
        else:
            self._is_built = False

        return normalized_document

    def add_documents(
        self,
        documents: Iterable[BM25Document | Any],
        *,
        overwrite: bool = False,
    ) -> int:
        added_count = 0

        for document in documents:
            self.add_document(
                document,
                overwrite=overwrite,
                rebuild=False,
            )
            added_count += 1

        self._rebuild_statistics()

        return added_count

    def remove_document(
        self,
        document_id: str,
        *,
        rebuild: bool = True,
    ) -> BM25Document:
        normalized_id = self._normalize_identifier(document_id)

        try:
            removed = self._documents.pop(normalized_id)
        except KeyError as exc:
            raise BM25RetrieverError(
                f"'{normalized_id}' kimliğine sahip belge bulunamadı.",
            ) from exc

        if rebuild:
            self._rebuild_statistics()
        else:
            self._is_built = False

        return removed

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        minimum_score: float | None = None,
        report_id: str | None = None,
        page_numbers: Sequence[int] = (),
        category: str | None = None,
        standard_codes: Sequence[str] = (),
        term_weights: Mapping[str, float] | None = None,
    ) -> list[BM25SearchResult]:
        self._ensure_index()

        cleaned_query = self._clean_text(query)

        if not cleaned_query:
            return []

        query_tokens = self.tokenize(cleaned_query)

        if not query_tokens:
            return []

        result_limit = top_k or self.config.top_k
        score_threshold = (
            self.config.minimum_score
            if minimum_score is None
            else minimum_score
        )

        if result_limit <= 0:
            return []

        if score_threshold < 0:
            raise ValueError("minimum_score negatif olamaz.")

        normalized_term_weights = {
            self.normalize_text(term): float(weight)
            for term, weight in (term_weights or {}).items()
            if self.normalize_text(term) and float(weight) > 0
        }

        # Katalog sorgularında aynı alias terimleri birkaç kez tekrar
        # edebilir. BM25'te bu tekrarlar belge skorunu yapay biçimde
        # katlamamalıdır; her benzersiz terimi bir kez değerlendir.
        query_term_frequency = Counter(
            {term: 1 for term in query_tokens}
        )
        results: list[BM25SearchResult] = []

        for document_id, document in self._documents.items():
            if not self._matches_filters(
                document,
                report_id=report_id,
                page_numbers=page_numbers,
                category=category,
                standard_codes=standard_codes,
            ):
                continue

            score = self._score_document(
                document_id=document_id,
                query_term_frequency=query_term_frequency,
                term_weights=normalized_term_weights,
            )

            if score < score_threshold:
                continue

            matched_terms = tuple(
                sorted(
                    set(query_tokens).intersection(
                        self._term_frequencies[document_id],
                    ),
                ),
            )

            matched_term_ratio = (
                len(matched_terms)
                / max(len(set(query_tokens)), 1)
            )

            # Tek bir genel terime dayanan sonuçları ele.
            if len(matched_terms) < 2 and matched_term_ratio < 0.15:
                continue

            results.append(
                BM25SearchResult(
                    document_id=document_id,
                    score=round(score, 6),
                    document=document,
                    matched_terms=matched_terms,
                    metadata={
                        "document_length": self._document_lengths[
                            document_id
                        ],
                        "matched_term_count": len(matched_terms),
                        "matched_term_ratio": round(
                            matched_term_ratio,
                            4,
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

        ranked_results = [
            BM25SearchResult(
                document_id=result.document_id,
                score=result.score,
                document=result.document,
                matched_terms=result.matched_terms,
                rank=index + 1,
                metadata=result.metadata,
            )
            for index, result in enumerate(results[:result_limit])
        ]

        return ranked_results

    def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[BM25SearchResult]:
        return self.retrieve(query, **kwargs)

    def get_document(
        self,
        document_id: str,
    ) -> BM25Document:
        normalized_id = self._normalize_identifier(document_id)

        try:
            return self._documents[normalized_id]
        except KeyError as exc:
            raise BM25RetrieverError(
                f"'{normalized_id}' kimliğine sahip belge bulunamadı.",
            ) from exc

    def all_documents(self) -> list[BM25Document]:
        return sorted(
            self._documents.values(),
            key=lambda document: (
                document.report_id or "",
                document.page_number
                if document.page_number is not None
                else 0,
                document.document_id,
            ),
        )

    def count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        self._documents.clear()
        self._document_tokens.clear()
        self._term_frequencies.clear()
        self._document_frequencies.clear()
        self._document_lengths.clear()
        self._average_document_length = 0.0
        self._is_built = False

    def tokenize(self, text: str) -> tuple[str, ...]:
        normalized = self.normalize_text(text)

        tokens = [
            token
            for token in self._TOKEN_PATTERN.findall(normalized)
            if len(token) >= self.config.minimum_token_length
        ]

        if self.config.remove_stopwords:
            stopwords = (
                self._TURKISH_STOPWORDS
                | self._ENGLISH_STOPWORDS
            )
            tokens = [
                token
                for token in tokens
                if token not in stopwords
            ]

        return tuple(tokens)

    def inverse_document_frequency(
        self,
        term: str,
    ) -> float:
        self._ensure_index()

        normalized_term = self.normalize_text(term)
        document_frequency = self._document_frequencies.get(
            normalized_term,
            0,
        )
        document_count = len(self._documents)

        if document_count == 0:
            return 0.0

        return math.log(
            1
            + (
                document_count
                - document_frequency
                + 0.5
            )
            / (
                document_frequency
                + 0.5
            ),
        )

    def _score_document(
        self,
        *,
        document_id: str,
        query_term_frequency: Counter[str],
        term_weights: Mapping[str, float],
    ) -> float:
        term_frequency = self._term_frequencies[document_id]
        document_length = self._document_lengths[document_id]

        score = 0.0

        for term, query_frequency in query_term_frequency.items():
            frequency = term_frequency.get(term, 0)

            if frequency == 0:
                continue

            idf = self.inverse_document_frequency(term)
            normalization = (
                frequency
                + self.config.k1
                * (
                    1
                    - self.config.b
                    + self.config.b
                    * document_length
                    / max(self._average_document_length, 1.0)
                )
            )

            term_score = (
                idf
                * frequency
                * (self.config.k1 + 1)
                / normalization
            )

            weight = term_weights.get(term, 1.0)

            score += term_score * weight * query_frequency

        return score

    def _rebuild_statistics(self) -> None:
        self._document_tokens.clear()
        self._term_frequencies.clear()
        self._document_frequencies.clear()
        self._document_lengths.clear()

        total_length = 0

        for document_id, document in self._documents.items():
            tokens = self.tokenize(document.text)
            term_frequency = Counter(tokens)

            self._document_tokens[document_id] = tokens
            self._term_frequencies[document_id] = term_frequency
            self._document_lengths[document_id] = len(tokens)

            total_length += len(tokens)

            for term in term_frequency:
                self._document_frequencies[term] += 1

        document_count = len(self._documents)

        self._average_document_length = (
            total_length / document_count
            if document_count > 0
            else 0.0
        )

        self._is_built = True

    def _matches_filters(
        self,
        document: BM25Document,
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
            and self.normalize_text(document.category or "")
            != self.normalize_text(category)
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

    def _load_repository_documents(
        self,
    ) -> Iterable[Any]:
        if self.repository is None:
            raise BM25RetrieverError(
                "İndeks oluşturmak için documents veya repository "
                "sağlanmalıdır.",
            )

        all_method = getattr(self.repository, "all", None)

        if callable(all_method):
            return all_method()

        raise BM25RetrieverError(
            "Repository all metodu sağlamıyor.",
        )

    def _to_document(
        self,
        source: BM25Document | Any,
    ) -> BM25Document:
        if isinstance(source, BM25Document):
            return source

        metadata = self._get_metadata(source)

        document_id = (
            metadata.get("original_chunk_uid")
            or metadata.get("chunk_id")
            or metadata.get("document_id")
            or metadata.get("identifier")
        )

        if document_id is None:
            document_id = self._get_first_value(
                source,
                (
                    "chunk_id",
                    "document_id",
                    "identifier",
                    "id",
                ),
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
            raise BM25RetrieverError(
                "Belge üzerinde chunk_id, document_id veya id bulunamadı.",
            )

        cleaned_text = self._clean_text(text)

        if not cleaned_text:
            raise BM25RetrieverError(
                f"'{document_id}' kimliğine sahip belgenin metni boş.",
            )

        report_id = (
            metadata.get("original_report_uid")
            or metadata.get("report_id")
        )

        if report_id is None:
            report_id = self._get_first_value(
                source,
                ("report_id",),
            )

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
            ("category",),
        )
        category = category or metadata.get("category")

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

        return BM25Document(
            document_id=self._normalize_identifier(document_id),
            text=cleaned_text,
            report_id=self._optional_string(report_id),
            page_number=self._optional_int(page_number),
            section_title=self._optional_string(section_title),
            category=self._optional_string(category),
            standard_codes=self._normalize_string_sequence(
                standard_codes,
            ),
            metadata=metadata,
        )

    def _ensure_index(self) -> None:
        if not self._is_built:
            raise BM25IndexNotBuiltError(
                "BM25 indeksi oluşturulmamış. Önce build_index çağrılmalıdır.",
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
                enum_value = getattr(value, "value", value)
                return enum_value

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

        normalized = {
            cls._clean_text(item)
            for item in iterable
            if cls._clean_text(item)
        }

        return tuple(sorted(normalized))

    @staticmethod
    def _normalize_standard_code(value: str) -> str:
        normalized = " ".join(
            str(value).strip().upper().split(),
        )

        normalized = re.sub(
            r"\s*[-:]\s*",
            " ",
            normalized,
            count=1,
        )

        return normalized

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""

        enum_value = getattr(value, "value", value)

        return " ".join(
            str(enum_value).replace("\n", " ").split(),
        )

    def normalize_text(self, value: str) -> str:
        if not value:
            return ""

        normalized = str(value)

        if self.config.lowercase:
            normalized = normalized.casefold()

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

        normalized = normalized.translate(translation_table)

        normalized = re.sub(
            r"[^\w\s\-./]",
            " ",
            normalized,
        )

        return " ".join(normalized.split())


__all__ = [
    "BM25Document",
    "BM25IndexNotBuiltError",
    "BM25Retriever",
    "BM25RetrieverConfig",
    "BM25RetrieverError",
    "BM25SearchResult",
    "ChunkRepositoryProtocol",
]