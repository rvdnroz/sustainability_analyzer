from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf  # type: ignore[no-redef]

try:
    import pytesseract
    from pytesseract import Output
except ImportError as exc:
    raise ImportError(
        "page_ocr.py için pytesseract paketi gereklidir. "
        "'python -m pip install pytesseract' komutunu çalıştırın."
    ) from exc

from .native_text_extractor import (
    NativePageText,
    TextBoundingBox,
)


logger = logging.getLogger(__name__)


class PageOCRError(RuntimeError):
    """Sayfa OCR işlemi tamamlanamadığında yükseltilir."""


@dataclass(slots=True, frozen=True)
class OCRPageWord:
    """OCR tarafından tespit edilen tek bir sözcük."""

    text: str
    normalized_text: str

    confidence: float

    bounding_box_pixels: tuple[int, int, int, int]
    bounding_box_pdf: TextBoundingBox

    block_number: int
    paragraph_number: int
    line_number: int
    word_number: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
            "bounding_box_pixels": list(
                self.bounding_box_pixels
            ),
            "bounding_box_pdf": (
                self.bounding_box_pdf.to_dict()
            ),
            "block_number": self.block_number,
            "paragraph_number": self.paragraph_number,
            "line_number": self.line_number,
            "word_number": self.word_number,
        }


@dataclass(slots=True, frozen=True)
class OCRPageLine:
    """OCR sözcüklerinden oluşturulan tek bir metin satırı."""

    line_index: int

    text: str
    normalized_text: str

    words: tuple[OCRPageWord, ...]

    confidence: float
    bounding_box_pdf: TextBoundingBox

    is_empty: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_index": self.line_index,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "words": [
                word.to_dict()
                for word in self.words
            ],
            "confidence": self.confidence,
            "bounding_box_pdf": (
                self.bounding_box_pdf.to_dict()
            ),
            "is_empty": self.is_empty,
        }


@dataclass(slots=True, frozen=True)
class OCRPageBlock:
    """OCR sonucundaki tek bir metin bloğu."""

    block_index: int

    text: str
    normalized_text: str

    lines: tuple[OCRPageLine, ...]

    confidence: float
    bounding_box_pdf: TextBoundingBox

    reading_order: int

    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "lines": [
                line.to_dict()
                for line in self.lines
            ],
            "confidence": self.confidence,
            "bounding_box_pdf": (
                self.bounding_box_pdf.to_dict()
            ),
            "reading_order": self.reading_order,
            "content_hash": self.content_hash,
        }


@dataclass(slots=True, frozen=True)
class OCRPageResult:
    """Tek bir PDF sayfasının OCR sonucu."""

    page_index: int
    page_number: int

    page_width: float
    page_height: float

    image_width: int
    image_height: int
    dpi: int

    text: str
    normalized_text: str

    blocks: tuple[OCRPageBlock, ...]
    lines: tuple[OCRPageLine, ...]
    words: tuple[OCRPageWord, ...]

    character_count: int
    word_count: int
    line_count: int
    block_count: int

    average_confidence: float
    minimum_confidence: float
    maximum_confidence: float

    language: str
    page_segmentation_mode: int

    is_valid: bool
    requires_vlm_fallback: bool

    rendered_image_path: Path | None
    content_hash: str

    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_number": self.page_number,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "dpi": self.dpi,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "blocks": [
                block.to_dict()
                for block in self.blocks
            ],
            "lines": [
                line.to_dict()
                for line in self.lines
            ],
            "words": [
                word.to_dict()
                for word in self.words
            ],
            "character_count": self.character_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "block_count": self.block_count,
            "average_confidence": (
                self.average_confidence
            ),
            "minimum_confidence": (
                self.minimum_confidence
            ),
            "maximum_confidence": (
                self.maximum_confidence
            ),
            "language": self.language,
            "page_segmentation_mode": (
                self.page_segmentation_mode
            ),
            "is_valid": self.is_valid,
            "requires_vlm_fallback": (
                self.requires_vlm_fallback
            ),
            "rendered_image_path": (
                str(self.rendered_image_path)
                if self.rendered_image_path is not None
                else None
            ),
            "content_hash": self.content_hash,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class OCRDocumentResult:
    """Bir PDF için gerçekleştirilen sayfa OCR işlemlerinin özeti."""

    pdf_path: Path

    processed_page_count: int
    successful_page_count: int
    failed_page_count: int

    total_character_count: int
    total_word_count: int

    pages: tuple[OCRPageResult, ...]

    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf_path": str(self.pdf_path),
            "processed_page_count": (
                self.processed_page_count
            ),
            "successful_page_count": (
                self.successful_page_count
            ),
            "failed_page_count": (
                self.failed_page_count
            ),
            "total_character_count": (
                self.total_character_count
            ),
            "total_word_count": (
                self.total_word_count
            ),
            "pages": [
                page.to_dict()
                for page in self.pages
            ],
            "warnings": list(self.warnings),
        }


class PageOCR:
    """
    Native metni yetersiz olan PDF sayfalarına OCR uygular.

    İşlem sırası:

    1. PDF sayfası belirlenen DPI ile görüntüye dönüştürülür.
    2. Görüntü gri tona çevrilir.
    3. Gürültü giderme, deskew ve threshold uygulanır.
    4. Tesseract image_to_data ile sözcükler çıkarılır.
    5. Sözcükler satır ve blok yapısında birleştirilir.
    6. OCR güven skorları hesaplanır.
    7. Kalitesi yetersiz sayfalar VLM fallback için işaretlenir.

    OCR yalnızca native metin bulunmayan veya yetersiz olan sayfalarda
    çalıştırılmalıdır.
    """

    _WHITESPACE_RE = re.compile(r"\s+")
    _MULTIPLE_NEWLINE_RE = re.compile(r"\n{3,}")

    def __init__(
        self,
        *,
        language: str = "tur+eng",
        tesseract_command: str | Path | None = None,
        dpi: int = 250,
        page_segmentation_mode: int = 3,
        oem: int = 3,
        minimum_word_confidence: float = 25.0,
        minimum_page_confidence: float = 45.0,
        minimum_character_count: int = 40,
        minimum_word_count: int = 8,
        vlm_fallback_confidence: float = 35.0,
        apply_denoising: bool = True,
        apply_threshold: bool = True,
        apply_deskew: bool = True,
        apply_sharpening: bool = True,
        save_rendered_images: bool = True,
        image_extension: str = ".png",
    ) -> None:
        if not language.strip():
            raise ValueError(
                "language boş olamaz."
            )

        if dpi < 72:
            raise ValueError(
                "dpi en az 72 olmalıdır."
            )

        if not 0 <= page_segmentation_mode <= 13:
            raise ValueError(
                "page_segmentation_mode 0 ile 13 arasında olmalıdır."
            )

        if not 0 <= oem <= 3:
            raise ValueError(
                "oem 0 ile 3 arasında olmalıdır."
            )

        for name, value in {
            "minimum_word_confidence": minimum_word_confidence,
            "minimum_page_confidence": minimum_page_confidence,
            "vlm_fallback_confidence": vlm_fallback_confidence,
        }.items():
            if not 0.0 <= value <= 100.0:
                raise ValueError(
                    f"{name} 0 ile 100 arasında olmalıdır."
                )

        if minimum_character_count < 0:
            raise ValueError(
                "minimum_character_count negatif olamaz."
            )

        if minimum_word_count < 0:
            raise ValueError(
                "minimum_word_count negatif olamaz."
            )

        normalized_extension = image_extension.lower()

        if not normalized_extension.startswith("."):
            normalized_extension = (
                f".{normalized_extension}"
            )

        if normalized_extension not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            raise ValueError(
                "Desteklenmeyen görüntü uzantısı: "
                f"{normalized_extension}"
            )

        self.language = language
        self.dpi = dpi
        self.page_segmentation_mode = (
            page_segmentation_mode
        )
        self.oem = oem

        self.minimum_word_confidence = (
            minimum_word_confidence
        )
        self.minimum_page_confidence = (
            minimum_page_confidence
        )
        self.minimum_character_count = (
            minimum_character_count
        )
        self.minimum_word_count = (
            minimum_word_count
        )
        self.vlm_fallback_confidence = (
            vlm_fallback_confidence
        )

        self.apply_denoising = apply_denoising
        self.apply_threshold = apply_threshold
        self.apply_deskew = apply_deskew
        self.apply_sharpening = apply_sharpening

        self.save_rendered_images = (
            save_rendered_images
        )
        self.image_extension = normalized_extension

        if tesseract_command is not None:
            pytesseract.pytesseract.tesseract_cmd = str(
                Path(
                    tesseract_command
                ).expanduser()
            )

        self._validate_tesseract()

    def process_pdf(
        self,
        pdf_path: str | Path,
        *,
        page_numbers: Iterable[int] | None = None,
        native_pages: Sequence[NativePageText] | None = None,
        output_dir: str | Path | None = None,
        only_pages_requiring_ocr: bool = True,
    ) -> OCRDocumentResult:
        """
        PDF içerisindeki seçili sayfalara OCR uygular.

        native_pages verilirse only_pages_requiring_ocr=True durumunda
        yalnızca needs_ocr=True olan sayfalar işlenir.
        """

        pdf_path = Path(
            pdf_path
        ).expanduser().resolve()

        self._validate_pdf_path(
            pdf_path
        )

        output_root = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else None
        )

        if output_root is not None:
            output_root.mkdir(
                parents=True,
                exist_ok=True,
            )

        native_page_map = {
            page.page_number: page
            for page in (
                native_pages or ()
            )
        }

        try:
            document = pymupdf.open(
                pdf_path
            )
        except Exception as exc:
            raise PageOCRError(
                f"PDF açılamadı: {pdf_path}"
            ) from exc

        try:
            if document.needs_pass:
                raise PageOCRError(
                    "Parola korumalı PDF OCR işlemine alınamaz."
                )

            selected_pages = self._normalize_pages(
                page_numbers=page_numbers,
                page_count=document.page_count,
            )

            if (
                only_pages_requiring_ocr
                and native_page_map
            ):
                selected_pages = [
                    page_number
                    for page_number in selected_pages
                    if (
                        page_number not in native_page_map
                        or native_page_map[
                            page_number
                        ].needs_ocr
                    )
                ]

            results: list[OCRPageResult] = []
            warnings: list[str] = []
            failed_count = 0

            for page_number in selected_pages:
                page = document.load_page(
                    page_number - 1
                )

                try:
                    result = self.process_page(
                        page,
                        output_dir=output_root,
                    )

                except Exception as exc:
                    failed_count += 1

                    message = (
                        f"Sayfa {page_number} OCR işlemi "
                        f"başarısız: {exc}"
                    )

                    warnings.append(message)
                    logger.warning(message)

                    continue

                results.append(result)

            return OCRDocumentResult(
                pdf_path=pdf_path,
                processed_page_count=len(
                    selected_pages
                ),
                successful_page_count=len(
                    results
                ),
                failed_page_count=failed_count,
                total_character_count=sum(
                    page.character_count
                    for page in results
                ),
                total_word_count=sum(
                    page.word_count
                    for page in results
                ),
                pages=tuple(results),
                warnings=tuple(warnings),
            )

        finally:
            document.close()

    def process_page(
        self,
        page: pymupdf.Page,
        *,
        output_dir: str | Path | None = None,
    ) -> OCRPageResult:
        """Açık bir PyMuPDF sayfasına OCR uygular."""

        page_index = int(
            page.number
        )
        page_number = page_index + 1

        page_width = float(
            page.rect.width
        )
        page_height = float(
            page.rect.height
        )

        image = self._render_page(
            page
        )

        image_height, image_width = (
            image.shape[:2]
        )

        processed_image = self._preprocess_image(
            image
        )

        rendered_image_path = (
            self._save_page_image(
                image=processed_image,
                output_dir=Path(
                    output_dir
                ).expanduser().resolve(),
                page_number=page_number,
            )
            if (
                output_dir is not None
                and self.save_rendered_images
            )
            else None
        )

        data = self._run_ocr(
            processed_image
        )

        words = self._build_words(
            data=data,
            page_rect=page.rect,
            image_width=image_width,
            image_height=image_height,
        )

        lines = self._build_lines(
            words
        )

        blocks = self._build_blocks(
            lines
        )

        text = self._join_blocks(
            blocks
        )

        normalized_text = (
            self._normalize_page_text(
                text
            )
        )

        confidences = [
            word.confidence
            for word in words
        ]

        average_confidence = (
            mean(confidences)
            if confidences
            else 0.0
        )

        minimum_confidence = (
            min(confidences)
            if confidences
            else 0.0
        )

        maximum_confidence = (
            max(confidences)
            if confidences
            else 0.0
        )

        character_count = len(
            normalized_text
        )

        word_count = len(
            normalized_text.split()
        )

        warnings: list[str] = []

        is_valid = (
            character_count
            >= self.minimum_character_count
            and word_count
            >= self.minimum_word_count
            and average_confidence
            >= self.minimum_page_confidence
        )

        requires_vlm_fallback = (
            average_confidence
            < self.vlm_fallback_confidence
            or character_count
            < self.minimum_character_count
            or word_count
            < self.minimum_word_count
        )

        if (
            average_confidence
            < self.minimum_page_confidence
        ):
            warnings.append(
                "Sayfa OCR güven skoru eşik altında: "
                f"{average_confidence:.2f}."
            )

        if (
            character_count
            < self.minimum_character_count
        ):
            warnings.append(
                "OCR karakter sayısı eşik altında: "
                f"{character_count}."
            )

        if word_count < self.minimum_word_count:
            warnings.append(
                "OCR sözcük sayısı eşik altında: "
                f"{word_count}."
            )

        if requires_vlm_fallback:
            warnings.append(
                "Sayfa VLM fallback için işaretlendi."
            )

        content_hash = hashlib.sha256(
            normalized_text.encode(
                "utf-8"
            )
        ).hexdigest()

        return OCRPageResult(
            page_index=page_index,
            page_number=page_number,
            page_width=round(
                page_width,
                3,
            ),
            page_height=round(
                page_height,
                3,
            ),
            image_width=image_width,
            image_height=image_height,
            dpi=self.dpi,
            text=text,
            normalized_text=normalized_text,
            blocks=tuple(blocks),
            lines=tuple(lines),
            words=tuple(words),
            character_count=character_count,
            word_count=word_count,
            line_count=len(lines),
            block_count=len(blocks),
            average_confidence=round(
                average_confidence,
                4,
            ),
            minimum_confidence=round(
                minimum_confidence,
                4,
            ),
            maximum_confidence=round(
                maximum_confidence,
                4,
            ),
            language=self.language,
            page_segmentation_mode=(
                self.page_segmentation_mode
            ),
            is_valid=is_valid,
            requires_vlm_fallback=(
                requires_vlm_fallback
            ),
            rendered_image_path=(
                rendered_image_path
            ),
            content_hash=content_hash,
            warnings=tuple(warnings),
            metadata={
                "ocr_engine": "tesseract",
                "oem": self.oem,
                "accepted_word_count": len(
                    words
                ),
                "minimum_word_confidence": (
                    self.minimum_word_confidence
                ),
            },
        )

    def _render_page(
        self,
        page: pymupdf.Page,
    ) -> np.ndarray:
        try:
            pixmap = page.get_pixmap(
                dpi=self.dpi,
                alpha=False,
                colorspace=pymupdf.csRGB,
            )

        except Exception as exc:
            raise PageOCRError(
                f"Sayfa {int(page.number) + 1} render edilemedi."
            ) from exc

        samples = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        )

        expected_size = (
            pixmap.height
            * pixmap.width
            * pixmap.n
        )

        if samples.size != expected_size:
            raise PageOCRError(
                "Render görüntüsü beklenen boyutta değil."
            )

        image = samples.reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )

        if pixmap.n == 3:
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        if pixmap.n == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR,
            )

        if pixmap.n == 1:
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

        raise PageOCRError(
            f"Desteklenmeyen kanal sayısı: {pixmap.n}"
        )

    def _preprocess_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        if self.apply_denoising:
            gray = cv2.fastNlMeansDenoising(
                gray,
                None,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        if self.apply_deskew:
            gray = self._deskew(
                gray
            )

        if self.apply_sharpening:
            kernel = np.array(
                [
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0],
                ],
                dtype=np.float32,
            )

            gray = cv2.filter2D(
                gray,
                -1,
                kernel,
            )

        if self.apply_threshold:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                15,
            )

        return gray

    def _run_ocr(
        self,
        image: np.ndarray,
    ) -> Mapping[str, Sequence[Any]]:
        config = (
            f"--oem {self.oem} "
            f"--psm {self.page_segmentation_mode} "
            "-c preserve_interword_spaces=1"
        )

        try:
            return pytesseract.image_to_data(
                image,
                lang=self.language,
                config=config,
                output_type=Output.DICT,
            )

        except pytesseract.TesseractNotFoundError as exc:
            raise PageOCRError(
                "Tesseract uygulaması bulunamadı."
            ) from exc

        except pytesseract.TesseractError as exc:
            raise PageOCRError(
                f"Tesseract OCR hatası: {exc}"
            ) from exc

    def _build_words(
        self,
        *,
        data: Mapping[str, Sequence[Any]],
        page_rect: pymupdf.Rect,
        image_width: int,
        image_height: int,
    ) -> list[OCRPageWord]:
        words: list[OCRPageWord] = []

        raw_texts = data.get(
            "text",
            ()
        )

        for index, raw_text in enumerate(
            raw_texts
        ):
            text = self._normalize_text(
                str(raw_text)
            )

            if not text:
                continue

            confidence = self._safe_float(
                self._data_value(
                    data,
                    "conf",
                    index,
                    -1,
                )
            )

            if confidence < self.minimum_word_confidence:
                continue

            left = self._safe_int(
                self._data_value(
                    data,
                    "left",
                    index,
                    0,
                )
            )

            top = self._safe_int(
                self._data_value(
                    data,
                    "top",
                    index,
                    0,
                )
            )

            width = self._safe_int(
                self._data_value(
                    data,
                    "width",
                    index,
                    0,
                )
            )

            height = self._safe_int(
                self._data_value(
                    data,
                    "height",
                    index,
                    0,
                )
            )

            pdf_box = self._pixel_box_to_pdf_box(
                left=left,
                top=top,
                width=width,
                height=height,
                page_rect=page_rect,
                image_width=image_width,
                image_height=image_height,
            )

            words.append(
                OCRPageWord(
                    text=text,
                    normalized_text=text,
                    confidence=round(
                        confidence,
                        4,
                    ),
                    bounding_box_pixels=(
                        left,
                        top,
                        width,
                        height,
                    ),
                    bounding_box_pdf=pdf_box,
                    block_number=self._safe_int(
                        self._data_value(
                            data,
                            "block_num",
                            index,
                            0,
                        )
                    ),
                    paragraph_number=self._safe_int(
                        self._data_value(
                            data,
                            "par_num",
                            index,
                            0,
                        )
                    ),
                    line_number=self._safe_int(
                        self._data_value(
                            data,
                            "line_num",
                            index,
                            0,
                        )
                    ),
                    word_number=self._safe_int(
                        self._data_value(
                            data,
                            "word_num",
                            index,
                            0,
                        )
                    ),
                )
            )

        return words

    def _build_lines(
        self,
        words: Sequence[OCRPageWord],
    ) -> list[OCRPageLine]:
        grouped: dict[
            tuple[int, int, int],
            list[OCRPageWord],
        ] = {}

        for word in words:
            key = (
                word.block_number,
                word.paragraph_number,
                word.line_number,
            )

            grouped.setdefault(
                key,
                [],
            ).append(word)

        lines: list[OCRPageLine] = []

        ordered_groups = sorted(
            grouped.values(),
            key=lambda group: (
                min(
                    word.bounding_box_pdf.y0
                    for word in group
                ),
                min(
                    word.bounding_box_pdf.x0
                    for word in group
                ),
            ),
        )

        for line_index, line_words in enumerate(
            ordered_groups
        ):
            ordered_words = sorted(
                line_words,
                key=lambda word: (
                    word.bounding_box_pdf.x0
                ),
            )

            text = " ".join(
                word.text
                for word in ordered_words
            )

            normalized_text = (
                self._normalize_text(
                    text
                )
            )

            confidence = mean(
                word.confidence
                for word in ordered_words
            )

            bounding_box = self._union_boxes(
                [
                    word.bounding_box_pdf
                    for word in ordered_words
                ]
            )

            lines.append(
                OCRPageLine(
                    line_index=line_index,
                    text=text,
                    normalized_text=(
                        normalized_text
                    ),
                    words=tuple(
                        ordered_words
                    ),
                    confidence=round(
                        confidence,
                        4,
                    ),
                    bounding_box_pdf=(
                        bounding_box
                    ),
                    is_empty=not bool(
                        normalized_text
                    ),
                )
            )

        return lines

    def _build_blocks(
        self,
        lines: Sequence[OCRPageLine],
    ) -> list[OCRPageBlock]:
        grouped: dict[
            int,
            list[OCRPageLine],
        ] = {}

        for line in lines:
            block_number = (
                line.words[0].block_number
                if line.words
                else 0
            )

            grouped.setdefault(
                block_number,
                [],
            ).append(line)

        blocks: list[OCRPageBlock] = []

        ordered_groups = sorted(
            grouped.values(),
            key=lambda group: (
                min(
                    line.bounding_box_pdf.y0
                    for line in group
                ),
                min(
                    line.bounding_box_pdf.x0
                    for line in group
                ),
            ),
        )

        for reading_order, block_lines in enumerate(
            ordered_groups
        ):
            ordered_lines = sorted(
                block_lines,
                key=lambda line: (
                    line.bounding_box_pdf.y0,
                    line.bounding_box_pdf.x0,
                ),
            )

            text = "\n".join(
                line.text
                for line in ordered_lines
                if line.text
            )

            normalized_text = (
                self._normalize_page_text(
                    text
                )
            )

            confidence = mean(
                line.confidence
                for line in ordered_lines
            )

            bounding_box = self._union_boxes(
                [
                    line.bounding_box_pdf
                    for line in ordered_lines
                ]
            )

            blocks.append(
                OCRPageBlock(
                    block_index=reading_order,
                    text=text,
                    normalized_text=(
                        normalized_text
                    ),
                    lines=tuple(
                        ordered_lines
                    ),
                    confidence=round(
                        confidence,
                        4,
                    ),
                    bounding_box_pdf=(
                        bounding_box
                    ),
                    reading_order=reading_order,
                    content_hash=hashlib.sha256(
                        normalized_text.encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                )
            )

        return blocks

    @staticmethod
    def _join_blocks(
        blocks: Sequence[OCRPageBlock],
    ) -> str:
        return "\n\n".join(
            block.text.strip()
            for block in blocks
            if block.text.strip()
        )

    @classmethod
    def _normalize_text(
        cls,
        text: str,
    ) -> str:
        text = text.replace(
            "\u00ad",
            ""
        )

        text = text.replace(
            "\r",
            " "
        )

        text = text.replace(
            "\n",
            " "
        )

        return cls._WHITESPACE_RE.sub(
            " ",
            text,
        ).strip()

    @classmethod
    def _normalize_page_text(
        cls,
        text: str,
    ) -> str:
        lines = [
            cls._normalize_text(line)
            for line in text.splitlines()
        ]

        normalized = "\n".join(
            line
            for line in lines
            if line
        )

        return cls._MULTIPLE_NEWLINE_RE.sub(
            "\n\n",
            normalized,
        ).strip()

    @staticmethod
    def _deskew(
        image: np.ndarray,
    ) -> np.ndarray:
        inverted = cv2.bitwise_not(
            image
        )

        coordinates = np.column_stack(
            np.where(
                inverted > 0
            )
        )

        if coordinates.size == 0:
            return image

        angle = cv2.minAreaRect(
            coordinates
        )[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.1:
            return image

        height, width = image.shape[:2]

        center = (
            width // 2,
            height // 2,
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )

        return cv2.warpAffine(
            image,
            matrix,
            (
                width,
                height,
            ),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _save_page_image(
        self,
        *,
        image: np.ndarray,
        output_dir: Path,
        page_number: int,
    ) -> Path:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        target_path = (
            output_dir
            / (
                f"page_{page_number:04d}"
                f"_ocr{self.image_extension}"
            )
        )

        success = cv2.imwrite(
            str(target_path),
            image,
        )

        if not success:
            raise PageOCRError(
                f"OCR sayfa görüntüsü yazılamadı: {target_path}"
            )

        return target_path

    @staticmethod
    def _pixel_box_to_pdf_box(
        *,
        left: int,
        top: int,
        width: int,
        height: int,
        page_rect: pymupdf.Rect,
        image_width: int,
        image_height: int,
    ) -> TextBoundingBox:
        scale_x = (
            float(page_rect.width)
            / max(
                image_width,
                1,
            )
        )

        scale_y = (
            float(page_rect.height)
            / max(
                image_height,
                1,
            )
        )

        return TextBoundingBox(
            x0=round(
                float(page_rect.x0)
                + left * scale_x,
                3,
            ),
            y0=round(
                float(page_rect.y0)
                + top * scale_y,
                3,
            ),
            x1=round(
                float(page_rect.x0)
                + (
                    left + width
                )
                * scale_x,
                3,
            ),
            y1=round(
                float(page_rect.y0)
                + (
                    top + height
                )
                * scale_y,
                3,
            ),
        )

    @staticmethod
    def _union_boxes(
        boxes: Sequence[TextBoundingBox],
    ) -> TextBoundingBox:
        if not boxes:
            raise ValueError(
                "Birleştirilecek bounding box bulunmuyor."
            )

        return TextBoundingBox(
            x0=min(
                box.x0
                for box in boxes
            ),
            y0=min(
                box.y0
                for box in boxes
            ),
            x1=max(
                box.x1
                for box in boxes
            ),
            y1=max(
                box.y1
                for box in boxes
            ),
        )

    def _validate_tesseract(
        self,
    ) -> None:
        try:
            version = (
                pytesseract.get_tesseract_version()
            )

        except pytesseract.TesseractNotFoundError as exc:
            raise PageOCRError(
                "Tesseract OCR sistemde bulunamadı."
            ) from exc

        logger.info(
            "Tesseract OCR hazır. Sürüm: %s",
            version,
        )

        try:
            installed_languages = set(
                pytesseract.get_languages(
                    config=""
                )
            )

        except pytesseract.TesseractError:
            installed_languages = set()

        requested_languages = set(
            self.language.split("+")
        )

        missing_languages = (
            requested_languages
            - installed_languages
        )

        if missing_languages:
            logger.warning(
                "Eksik Tesseract dil paketleri: %s",
                sorted(
                    missing_languages
                ),
            )

    @staticmethod
    def _data_value(
        data: Mapping[str, Sequence[Any]],
        key: str,
        index: int,
        default: Any,
    ) -> Any:
        values = data.get(
            key,
            ()
        )

        if index >= len(values):
            return default

        return values[index]

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float:
        try:
            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return -1.0

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int:
        try:
            return int(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0

    @staticmethod
    def _normalize_pages(
        *,
        page_numbers: Iterable[int] | None,
        page_count: int,
    ) -> list[int]:
        if page_numbers is None:
            return list(
                range(
                    1,
                    page_count + 1,
                )
            )

        pages = list(
            dict.fromkeys(
                int(page_number)
                for page_number in page_numbers
            )
        )

        invalid_pages = [
            page_number
            for page_number in pages
            if not 1
            <= page_number
            <= page_count
        ]

        if invalid_pages:
            raise ValueError(
                "Geçersiz sayfa numaraları: "
                f"{invalid_pages}. "
                f"Geçerli aralık 1-{page_count}."
            )

        return pages

    @staticmethod
    def _validate_pdf_path(
        pdf_path: Path,
    ) -> None:
        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF bulunamadı: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Yalnızca .pdf uzantılı dosyalar desteklenir."
            )


def run_page_ocr(
    pdf_path: str | Path,
    *,
    page_numbers: Iterable[int] | None = None,
    native_pages: Sequence[NativePageText] | None = None,
    output_dir: str | Path | None = None,
    language: str = "tur+eng",
    tesseract_command: str | Path | None = None,
) -> OCRDocumentResult:
    """PageOCR sınıfı için kısa kullanım arayüzü."""

    page_ocr = PageOCR(
        language=language,
        tesseract_command=tesseract_command,
    )

    return page_ocr.process_pdf(
        pdf_path=pdf_path,
        page_numbers=page_numbers,
        native_pages=native_pages,
        output_dir=output_dir,
        only_pages_requiring_ocr=True,
    )