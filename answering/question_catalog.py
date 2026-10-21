from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence


_ALLOWED_ESG_CATEGORIES = frozenset(
    {
        "environmental",
        "social",
        "governance",
        "economic",
        "standards",
        "general",
    }
)

_ALLOWED_ANSWER_TYPES = frozenset(
    {
        "number",
        "percentage",
        "currency",
        "boolean",
        "text",
        "list",
        "comparison",
        "trend",
        "target",
        "unknown",
    }
)

_ALLOWED_RETRIEVAL_STRATEGIES = frozenset(
    {
        "table_first",
        "text_first",
        "standard_first",
        "hybrid",
    }
)

_ALLOWED_PREFERRED_SOURCES = frozenset(
    {
        "table",
        "text",
        "standard",
    }
)

_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_EXCEL_COLUMN_PATTERN = re.compile(r"^[A-Z]+$")


class QuestionCatalogError(RuntimeError):
    """Soru kataloğunun okunması veya doğrulanması sırasında oluşan hata."""


class QuestionCatalogValidationError(QuestionCatalogError, ValueError):
    """Soru kataloğunda geçersiz ya da tutarsız veri bulunduğunu belirtir."""


@dataclass(slots=True, frozen=True)
class QuestionDefinition:
    """Tek bir sabit sürdürülebilirlik sorusunun veri sözleşmesi."""

    id: int
    key: str
    question: str
    source_column: str
    original_header: str

    group: str
    group_order: int
    esg_category: str
    topic: str

    expected_answer_type: str
    retrieval_strategy: str

    aliases: tuple[str, ...] = ()
    expected_units: tuple[str, ...] = ()
    preferred_sources: tuple[str, ...] = ()
    preferred_document_types: tuple[str, ...] = ()

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id <= 0:
            raise QuestionCatalogValidationError(
                f"Soru id değeri pozitif olmalıdır: {self.id!r}."
            )

        self._validate_identifier("key", self.key)
        self._validate_identifier("group", self.group)
        self._validate_identifier("topic", self.topic)

        if not self.question.strip():
            raise QuestionCatalogValidationError(
                f"Soru {self.id}: question boş olamaz."
            )

        if not self.original_header.strip():
            raise QuestionCatalogValidationError(
                f"Soru {self.id}: original_header boş olamaz."
            )

        if not _EXCEL_COLUMN_PATTERN.fullmatch(self.source_column):
            raise QuestionCatalogValidationError(
                f"Soru {self.id}: geçersiz Excel sütunu: "
                f"{self.source_column!r}."
            )

        if self.group_order <= 0:
            raise QuestionCatalogValidationError(
                f"Soru {self.id}: group_order pozitif olmalıdır."
            )

        if self.esg_category not in _ALLOWED_ESG_CATEGORIES:
            allowed = ", ".join(sorted(_ALLOWED_ESG_CATEGORIES))
            raise QuestionCatalogValidationError(
                f"Soru {self.id}: geçersiz esg_category "
                f"{self.esg_category!r}. İzin verilenler: {allowed}."
            )

        if self.expected_answer_type not in _ALLOWED_ANSWER_TYPES:
            allowed = ", ".join(sorted(_ALLOWED_ANSWER_TYPES))
            raise QuestionCatalogValidationError(
                f"Soru {self.id}: geçersiz expected_answer_type "
                f"{self.expected_answer_type!r}. İzin verilenler: {allowed}."
            )

        if self.retrieval_strategy not in _ALLOWED_RETRIEVAL_STRATEGIES:
            allowed = ", ".join(sorted(_ALLOWED_RETRIEVAL_STRATEGIES))
            raise QuestionCatalogValidationError(
                f"Soru {self.id}: geçersiz retrieval_strategy "
                f"{self.retrieval_strategy!r}. İzin verilenler: {allowed}."
            )

        invalid_sources = sorted(
            set(self.preferred_sources) - _ALLOWED_PREFERRED_SOURCES
        )
        if invalid_sources:
            raise QuestionCatalogValidationError(
                f"Soru {self.id}: geçersiz preferred_sources değerleri: "
                f"{', '.join(invalid_sources)}."
            )

        if not self.aliases:
            raise QuestionCatalogValidationError(
                f"Soru {self.id}: aliases en az bir değer içermelidir."
            )

        if not self.preferred_sources:
            raise QuestionCatalogValidationError(
                f"Soru {self.id}: preferred_sources boş olamaz."
            )

        if not self.preferred_document_types:
            raise QuestionCatalogValidationError(
                f"Soru {self.id}: preferred_document_types boş olamaz."
            )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        item_index: int | None = None,
    ) -> QuestionDefinition:
        """Bir JSON nesnesini doğrulanmış soru tanımına dönüştürür."""

        if not isinstance(payload, Mapping):
            location = (
                f"{item_index}. öğe"
                if item_index is not None
                else "Soru kaydı"
            )
            raise QuestionCatalogValidationError(
                f"{location} bir JSON nesnesi olmalıdır."
            )

        required_fields = (
            "id",
            "key",
            "question",
            "source_column",
            "original_header",
            "group",
            "group_order",
            "esg_category",
            "topic",
            "expected_answer_type",
            "retrieval_strategy",
            "aliases",
            "expected_units",
            "preferred_sources",
            "preferred_document_types",
        )
        missing = [name for name in required_fields if name not in payload]
        if missing:
            location = (
                f"{item_index}. öğede"
                if item_index is not None
                else "Soru kaydında"
            )
            raise QuestionCatalogValidationError(
                f"{location} zorunlu alanlar eksik: {', '.join(missing)}."
            )

        known_fields = set(required_fields)
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in known_fields
        }

        try:
            question_id = int(payload["id"])
            group_order = int(payload["group_order"])
        except (TypeError, ValueError) as exc:
            raise QuestionCatalogValidationError(
                "id ve group_order tam sayı olmalıdır."
            ) from exc

        return cls(
            id=question_id,
            key=_required_string(payload, "key"),
            question=_required_string(payload, "question"),
            source_column=_required_string(payload, "source_column").upper(),
            original_header=_required_string(payload, "original_header"),
            group=_required_string(payload, "group"),
            group_order=group_order,
            esg_category=_required_string(payload, "esg_category"),
            topic=_required_string(payload, "topic"),
            expected_answer_type=_required_string(
                payload,
                "expected_answer_type",
            ),
            retrieval_strategy=_required_string(
                payload,
                "retrieval_strategy",
            ),
            aliases=_string_tuple(payload, "aliases"),
            expected_units=_string_tuple(payload, "expected_units"),
            preferred_sources=_string_tuple(payload, "preferred_sources"),
            preferred_document_types=_string_tuple(
                payload,
                "preferred_document_types",
            ),
            metadata=metadata,
        )

    @property
    def retrieval_terms(self) -> tuple[str, ...]:
        """Soru metni, anahtar ve alias değerlerinden tekil arama terimleri."""

        return _deduplicate_strings(
            (
                self.question,
                self.key.replace("_", " "),
                *self.aliases,
            )
        )

    def build_retrieval_query(self) -> str:
        """Retriever bileşenlerine verilebilecek genişletilmiş sorguyu üretir."""

        return " | ".join(self.retrieval_terms)

    def to_mapping(self) -> dict[str, Any]:
        """Soru tanımını JSON uyumlu sözlüğe dönüştürür."""

        result: dict[str, Any] = {
            "id": self.id,
            "key": self.key,
            "question": self.question,
            "source_column": self.source_column,
            "original_header": self.original_header,
            "group": self.group,
            "group_order": self.group_order,
            "esg_category": self.esg_category,
            "topic": self.topic,
            "expected_answer_type": self.expected_answer_type,
            "retrieval_strategy": self.retrieval_strategy,
            "aliases": list(self.aliases),
            "expected_units": list(self.expected_units),
            "preferred_sources": list(self.preferred_sources),
            "preferred_document_types": list(
                self.preferred_document_types
            ),
        }
        result.update(dict(self.metadata))
        return result

    @staticmethod
    def _validate_identifier(field_name: str, value: str) -> None:
        if not _KEY_PATTERN.fullmatch(value):
            raise QuestionCatalogValidationError(
                f"{field_name} yalnızca küçük harf, sayı ve alt çizgi "
                f"içerebilir; harfle başlamalıdır: {value!r}."
            )


@dataclass(slots=True, frozen=True)
class QuestionGroup:
    """Aynı model çağrısında cevaplanacak soru grubunu temsil eder."""

    name: str
    order: int
    questions: tuple[QuestionDefinition, ...]
    esg_categories: tuple[str, ...]
    topics: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.questions:
            raise QuestionCatalogValidationError(
                f"{self.name!r} soru grubu boş olamaz."
            )

        for question in self.questions:
            if question.group != self.name:
                raise QuestionCatalogValidationError(
                    f"Soru {question.id}, {self.name!r} yerine "
                    f"{question.group!r} grubuna bağlı."
                )
            if question.group_order != self.order:
                raise QuestionCatalogValidationError(
                    f"Soru {question.id} için group_order "
                    f"{question.group_order}; grup sırası {self.order}."
                )

    @property
    def question_ids(self) -> tuple[int, ...]:
        return tuple(question.id for question in self.questions)

    @property
    def size(self) -> int:
        return len(self.questions)


class QuestionCatalog:
    """Sabit soruları yükler, doğrular ve kategori bazında gruplar."""

    def __init__(
        self,
        questions: Sequence[QuestionDefinition],
        *,
        expected_question_count: int | None = 41,
        require_contiguous_ids: bool = True,
    ) -> None:
        ordered_questions = tuple(sorted(questions, key=lambda item: item.id))
        if not ordered_questions:
            raise QuestionCatalogValidationError(
                "Soru kataloğu en az bir soru içermelidir."
            )

        self._validate_catalog(
            ordered_questions,
            expected_question_count=expected_question_count,
            require_contiguous_ids=require_contiguous_ids,
        )

        self._questions = ordered_questions
        self._by_id = {
            question.id: question for question in ordered_questions
        }
        self._by_key = {
            question.key: question for question in ordered_questions
        }
        self._by_source_column = {
            question.source_column: question
            for question in ordered_questions
        }
        self._groups = self._build_groups(ordered_questions)
        self._groups_by_name = {
            group.name: group for group in self._groups
        }

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_question_count: int | None = 41,
        require_contiguous_ids: bool = True,
    ) -> QuestionCatalog:
        source_path = Path(path).expanduser()
        if not source_path.is_file():
            raise QuestionCatalogError(
                f"Soru kataloğu bulunamadı: {source_path}."
            )

        try:
            text = source_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise QuestionCatalogError(
                f"Soru kataloğu okunamadı: {source_path}."
            ) from exc

        return cls.from_json(
            text,
            expected_question_count=expected_question_count,
            require_contiguous_ids=require_contiguous_ids,
        )

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        expected_question_count: int | None = 41,
        require_contiguous_ids: bool = True,
    ) -> QuestionCatalog:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise QuestionCatalogError(
                f"questions.json geçerli JSON değil: "
                f"satır {exc.lineno}, sütun {exc.colno}."
            ) from exc

        if not isinstance(payload, list):
            raise QuestionCatalogValidationError(
                "questions.json kök değeri bir liste olmalıdır."
            )

        questions = tuple(
            QuestionDefinition.from_mapping(item, item_index=index)
            for index, item in enumerate(payload, start=1)
        )
        return cls(
            questions,
            expected_question_count=expected_question_count,
            require_contiguous_ids=require_contiguous_ids,
        )

    @property
    def questions(self) -> tuple[QuestionDefinition, ...]:
        return self._questions

    @property
    def groups(self) -> tuple[QuestionGroup, ...]:
        return self._groups

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(group.name for group in self._groups)

    def __len__(self) -> int:
        return len(self._questions)

    def __iter__(self) -> Iterator[QuestionDefinition]:
        return iter(self._questions)

    def get_by_id(self, question_id: int) -> QuestionDefinition:
        try:
            return self._by_id[int(question_id)]
        except (KeyError, TypeError, ValueError) as exc:
            raise QuestionCatalogError(
                f"Soru id bulunamadı: {question_id!r}."
            ) from exc

    def get_by_key(self, key: str) -> QuestionDefinition:
        normalized_key = str(key).strip()
        try:
            return self._by_key[normalized_key]
        except KeyError as exc:
            raise QuestionCatalogError(
                f"Soru anahtarı bulunamadı: {normalized_key!r}."
            ) from exc

    def get_by_source_column(self, source_column: str) -> QuestionDefinition:
        normalized_column = str(source_column).strip().upper()
        try:
            return self._by_source_column[normalized_column]
        except KeyError as exc:
            raise QuestionCatalogError(
                f"Excel sütunu için soru bulunamadı: "
                f"{normalized_column!r}."
            ) from exc

    def get_group(self, group_name: str) -> QuestionGroup:
        normalized_name = str(group_name).strip()
        try:
            return self._groups_by_name[normalized_name]
        except KeyError as exc:
            raise QuestionCatalogError(
                f"Soru grubu bulunamadı: {normalized_name!r}."
            ) from exc

    def questions_for_group(
        self,
        group_name: str,
    ) -> tuple[QuestionDefinition, ...]:
        return self.get_group(group_name).questions

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            [question.to_mapping() for question in self._questions],
            ensure_ascii=False,
            indent=indent,
        )

    @staticmethod
    def _validate_catalog(
        questions: Sequence[QuestionDefinition],
        *,
        expected_question_count: int | None,
        require_contiguous_ids: bool,
    ) -> None:
        if (
            expected_question_count is not None
            and len(questions) != expected_question_count
        ):
            raise QuestionCatalogValidationError(
                f"Soru kataloğunda {expected_question_count} soru "
                f"bekleniyordu; {len(questions)} soru bulundu."
            )

        QuestionCatalog._ensure_unique(
            questions,
            field_name="id",
            value_getter=lambda item: item.id,
        )
        QuestionCatalog._ensure_unique(
            questions,
            field_name="key",
            value_getter=lambda item: item.key,
        )
        QuestionCatalog._ensure_unique(
            questions,
            field_name="source_column",
            value_getter=lambda item: item.source_column,
        )

        if require_contiguous_ids:
            ids = sorted(question.id for question in questions)
            expected_ids = list(range(ids[0], ids[-1] + 1))
            if ids != expected_ids:
                missing_ids = sorted(set(expected_ids) - set(ids))
                raise QuestionCatalogValidationError(
                    "Soru id değerleri kesintisiz olmalıdır. Eksik id "
                    f"değerleri: {missing_ids}."
                )

        group_to_order: dict[str, int] = {}
        order_to_group: dict[int, str] = {}
        for question in questions:
            existing_order = group_to_order.setdefault(
                question.group,
                question.group_order,
            )
            if existing_order != question.group_order:
                raise QuestionCatalogValidationError(
                    f"{question.group!r} grubu birden fazla group_order "
                    "değerine sahip."
                )

            existing_group = order_to_group.setdefault(
                question.group_order,
                question.group,
            )
            if existing_group != question.group:
                raise QuestionCatalogValidationError(
                    f"group_order={question.group_order} hem "
                    f"{existing_group!r} hem {question.group!r} için "
                    "kullanılmış."
                )

        group_orders = sorted(order_to_group)
        expected_orders = list(range(1, len(group_orders) + 1))
        if group_orders != expected_orders:
            raise QuestionCatalogValidationError(
                "Grup sıraları 1'den başlayarak kesintisiz olmalıdır. "
                f"Bulunan sıralar: {group_orders}."
            )

    @staticmethod
    def _build_groups(
        questions: Sequence[QuestionDefinition],
    ) -> tuple[QuestionGroup, ...]:
        grouped: dict[str, list[QuestionDefinition]] = {}
        group_orders: dict[str, int] = {}

        for question in questions:
            grouped.setdefault(question.group, []).append(question)
            group_orders[question.group] = question.group_order

        result: list[QuestionGroup] = []
        for group_name, group_questions in grouped.items():
            ordered = tuple(sorted(group_questions, key=lambda item: item.id))
            result.append(
                QuestionGroup(
                    name=group_name,
                    order=group_orders[group_name],
                    questions=ordered,
                    esg_categories=_deduplicate_strings(
                        question.esg_category for question in ordered
                    ),
                    topics=_deduplicate_strings(
                        question.topic for question in ordered
                    ),
                )
            )

        return tuple(sorted(result, key=lambda item: item.order))

    @staticmethod
    def _ensure_unique(
        questions: Sequence[QuestionDefinition],
        *,
        field_name: str,
        value_getter: Any,
    ) -> None:
        seen: dict[Any, int] = {}
        duplicates: list[str] = []

        for question in questions:
            value = value_getter(question)
            if value in seen:
                duplicates.append(
                    f"{value!r} (sorular {seen[value]} ve {question.id})"
                )
            else:
                seen[value] = question.id

        if duplicates:
            raise QuestionCatalogValidationError(
                f"Tekrar eden {field_name} değerleri: "
                + ", ".join(duplicates)
                + "."
            )


def load_question_catalog(
    path: str | Path,
    *,
    expected_question_count: int | None = 41,
) -> QuestionCatalog:
    """Soru kataloğunu yüklemek için kısa yol fonksiyonu."""

    return QuestionCatalog.from_file(
        path,
        expected_question_count=expected_question_count,
    )


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise QuestionCatalogValidationError(
            f"{field_name} boş olmayan bir metin olmalıdır."
        )
    return value.strip()


def _string_tuple(
    payload: Mapping[str, Any],
    field_name: str,
) -> tuple[str, ...]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise QuestionCatalogValidationError(
            f"{field_name} bir JSON listesi olmalıdır."
        )

    items: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str) or not item.strip():
            raise QuestionCatalogValidationError(
                f"{field_name}[{index}] boş olmayan bir metin olmalıdır."
            )
        items.append(item.strip())

    return _deduplicate_strings(items)


def _deduplicate_strings(values: Sequence[str] | Any) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = str(value).strip()
        if not normalized:
            continue
        identity = normalized.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        result.append(normalized)

    return tuple(result)


__all__ = [
    "QuestionCatalog",
    "QuestionCatalogError",
    "QuestionCatalogValidationError",
    "QuestionDefinition",
    "QuestionGroup",
    "load_question_catalog",
]
