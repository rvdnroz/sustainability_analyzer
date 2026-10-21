from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class PageType(str, Enum):
    """
    PDF sayfasının içerik türünü belirtir.

    Bir sayfa birden fazla içerik türüne sahip olabilir.
    Örneğin aynı sayfada hem metin hem tablo bulunabilir.
    """

    TEXT = "text"
    TABLE = "table"
    GRAPHIC = "graphic"
    STANDARD_INDEX = "standard_index"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LayoutElementType(str, Enum):
    """
    Sayfa üzerindeki yapısal eleman türlerini belirtir.
    """

    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    LIST = "list"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    OTHER = "other"


class BoundingBox(BaseModel):
    """
    Sayfa üzerindeki bir elemanın koordinatlarını temsil eder.

    Koordinat sistemi:

        (x1, y1) --------
           |            |
           |            |
           -------- (x2, y2)

    Koordinatlar piksel veya PDF koordinatı olabilir.
    Kullanılan koordinat sistemi metadata alanında ayrıca belirtilebilir.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    x1: float = Field(
        ge=0.0,
    )

    y1: float = Field(
        ge=0.0,
    )

    x2: float = Field(
        ge=0.0,
    )

    y2: float = Field(
        ge=0.0,
    )

    @model_validator(mode="after")
    def validate_coordinates(
        self,
    ) -> BoundingBox:
        """
        Sağ-alt koordinatların sol-üst koordinatlardan büyük olmasını
        kontrol eder.
        """

        if self.x2 <= self.x1:
            raise ValueError(
                "BoundingBox için x2 değeri x1 değerinden büyük olmalıdır."
            )

        if self.y2 <= self.y1:
            raise ValueError(
                "BoundingBox için y2 değeri y1 değerinden büyük olmalıdır."
            )

        return self

    @property
    def width(self) -> float:
        """
        Sınır kutusunun genişliğini döndürür.
        """

        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """
        Sınır kutusunun yüksekliğini döndürür.
        """

        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """
        Sınır kutusunun alanını döndürür.
        """

        return self.width * self.height


class LayoutElement(BaseModel):
    """
    Sayfa üzerinde tespit edilen tek bir yapısal elemanı temsil eder.

    Örnek elemanlar:

    - Başlık
    - Paragraf
    - Tablo
    - Grafik
    - Şekil açıklaması
    - Sayfa numarası
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    element_type: LayoutElementType

    bounding_box: BoundingBox

    text: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    reading_order: int | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("text")
    @classmethod
    def normalize_text(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Boş metin değerlerini None değerine dönüştürür.
        """

        if value is None:
            return None

        normalized_text = value.strip()

        if not normalized_text:
            return None

        return normalized_text


class Page(BaseModel):
    """
    Bir sürdürülebilirlik raporundaki tek bir PDF sayfasını temsil eder.

    Bu model şu modüller tarafından kullanılacaktır:

    - page_renderer.py
    - page_classifier.py
    - layout_detector.py
    - table_detector.py
    - section_detector.py
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    report_id: UUID

    page_number: int = Field(
        ge=1,
    )

    text: str = ""

    rendered_image_path: Path | None = None

    width: float | None = Field(
        default=None,
        gt=0.0,
    )

    height: float | None = Field(
        default=None,
        gt=0.0,
    )

    detected_types: list[PageType] = Field(
        default_factory=lambda: [PageType.UNKNOWN],
    )

    primary_type: PageType = PageType.UNKNOWN

    layout_elements: list[LayoutElement] = Field(
        default_factory=list,
    )

    has_native_text: bool = False

    needs_ocr: bool = False

    ocr_completed: bool = False

    classification_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("text")
    @classmethod
    def normalize_page_text(
        cls,
        value: str,
    ) -> str:
        """
        Sayfa metninin başındaki ve sonundaki gereksiz boşlukları temizler.
        """

        return value.strip()

    @field_validator("detected_types")
    @classmethod
    def validate_detected_types(
        cls,
        value: list[PageType],
    ) -> list[PageType]:
        """
        İçerik türü listesinin boş olmasını engeller ve tekrarları kaldırır.
        """

        if not value:
            return [PageType.UNKNOWN]

        unique_types: list[PageType] = []

        for page_type in value:
            if page_type not in unique_types:
                unique_types.append(page_type)

        return unique_types

    @model_validator(mode="after")
    def validate_page_classification(
        self,
    ) -> Page:
        """
        Birincil sayfa türünün detected_types listesinde bulunmasını sağlar.
        """

        if self.primary_type not in self.detected_types:
            self.detected_types.append(self.primary_type)

        if len(self.detected_types) > 1:
            non_unknown_types = [
                page_type
                for page_type in self.detected_types
                if page_type != PageType.UNKNOWN
            ]

            if non_unknown_types:
                self.detected_types = non_unknown_types

        return self

    @property
    def text_length(self) -> int:
        """
        Sayfadaki çıkarılmış metnin karakter sayısını döndürür.
        """

        return len(self.text)

    @property
    def has_text(self) -> bool:
        """
        Sayfada kullanılabilir metin olup olmadığını belirtir.
        """

        return bool(self.text.strip())

    @property
    def has_tables(self) -> bool:
        """
        Sayfanın tablo içerip içermediğini belirtir.
        """

        if PageType.TABLE in self.detected_types:
            return True

        return any(
            element.element_type == LayoutElementType.TABLE
            for element in self.layout_elements
        )

    @property
    def has_graphics(self) -> bool:
        """
        Sayfanın grafik veya şekil içerip içermediğini belirtir.
        """

        if PageType.GRAPHIC in self.detected_types:
            return True

        graphic_types = {
            LayoutElementType.FIGURE,
        }

        return any(
            element.element_type in graphic_types
            for element in self.layout_elements
        )

    @property
    def is_mixed(self) -> bool:
        """
        Sayfanın birden fazla içerik türü barındırıp barındırmadığını belirtir.
        """

        if PageType.MIXED in self.detected_types:
            return True

        meaningful_types = {
            page_type
            for page_type in self.detected_types
            if page_type not in {
                PageType.UNKNOWN,
                PageType.MIXED,
            }
        }

        return len(meaningful_types) > 1

    def set_classification(
        self,
        detected_types: list[PageType],
        primary_type: PageType,
        confidence: float,
    ) -> None:
        """
        page_classifier.py sonucunu sayfa nesnesine kaydeder.
        """

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "Sınıflandırma güven skoru 0 ile 1 arasında olmalıdır."
            )

        unique_types: list[PageType] = []

        for page_type in detected_types:
            if page_type not in unique_types:
                unique_types.append(page_type)

        if primary_type not in unique_types:
            unique_types.append(primary_type)

        non_unknown_types = [
            page_type
            for page_type in unique_types
            if page_type != PageType.UNKNOWN
        ]

        if non_unknown_types:
            unique_types = non_unknown_types

        if len(unique_types) > 1 and PageType.MIXED not in unique_types:
            unique_types.append(PageType.MIXED)

        self.detected_types = unique_types
        self.primary_type = primary_type
        self.classification_confidence = confidence

    def add_layout_element(
        self,
        element: LayoutElement,
    ) -> None:
        """
        Sayfaya yeni bir layout elemanı ekler.
        """

        self.layout_elements.append(element)

        self.layout_elements.sort(
            key=lambda item: (
                item.reading_order is None,
                item.reading_order
                if item.reading_order is not None
                else 0,
            )
        )

    def get_elements_by_type(
        self,
        element_type: LayoutElementType,
    ) -> list[LayoutElement]:
        """
        Belirli türdeki layout elemanlarını döndürür.
        """

        return [
            element
            for element in self.layout_elements
            if element.element_type == element_type
        ]

    def mark_ocr_required(self) -> None:
        """
        Sayfanın OCR işlemine ihtiyaç duyduğunu işaretler.
        """

        self.needs_ocr = True
        self.ocr_completed = False

    def mark_ocr_completed(
        self,
        extracted_text: str,
    ) -> None:
        """
        OCR işlemi tamamlandığında çıkarılan metni sayfaya kaydeder.
        """

        normalized_text = extracted_text.strip()

        if not normalized_text:
            raise ValueError(
                "OCR sonucunda elde edilen metin boş olamaz."
            )

        self.text = normalized_text
        self.needs_ocr = False
        self.ocr_completed = True

    def set_rendered_image(
        self,
        image_path: Path,
        width: float,
        height: float,
    ) -> None:
        """
        page_renderer.py tarafından oluşturulan sayfa görüntüsünü kaydeder.
        """

        if width <= 0 or height <= 0:
            raise ValueError(
                "Sayfa görüntüsünün genişlik ve yüksekliği pozitif olmalıdır."
            )

        self.rendered_image_path = image_path
        self.width = width
        self.height = height