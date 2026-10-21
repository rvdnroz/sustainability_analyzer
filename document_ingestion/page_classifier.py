from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import pymupdf
except ImportError:  # Eski PyMuPDF sürümleri için
    import fitz as pymupdf  # type: ignore[no-redef]

try:
    from models.page import PageType
except ImportError:  # Paket olarak çalıştırıldığında
    from ..models.page import PageType


logger = logging.getLogger(__name__)


class PageClassificationError(RuntimeError):
    """Bir PDF sayfası sınıflandırılamadığında yükseltilir."""


@dataclass(slots=True, frozen=True)
class PageClassification:
    """Tek bir PDF sayfasına ait sınıflandırma sonucu."""

    page_index: int
    page_number: int

    primary_type: PageType
    detected_types: tuple[PageType, ...]

    confidence: float

    has_native_text: bool
    needs_ocr: bool
    is_scanned: bool
    used_ocr_text: bool

    text: str

    width_points: float
    height_points: float
    rotation: int

    scores: dict[str, float]
    signals: dict[str, int | float]
    reasons: tuple[str, ...]

    # Render aşamasından gelen dosya bağlamı. Bu alanlar opsiyoneldir;
    # doğrudan açık bir PyMuPDF sayfası sınıflandırıldığında boş kalabilir.
    source_pdf: Path | None = None
    image_path: Path | None = None

    def to_dict(
        self,
        *,
        include_text: bool = False,
    ) -> dict[str, object]:
        data = asdict(self)

        data["primary_type"] = self.primary_type.value
        data["detected_types"] = [
            item.value for item in self.detected_types
        ]

        if self.source_pdf is not None:
            data["source_pdf"] = str(self.source_pdf)

        if self.image_path is not None:
            data["image_path"] = str(self.image_path)

        if not include_text:
            data.pop("text")
            data["text_preview"] = self.text[:500]

        return data


class PageClassifier:
    """
    PDF sayfalarını aşağıdaki türlerden biriyle sınıflandırır:

    - TEXT
    - TABLE
    - GRAPHIC
    - STANDARD_INDEX
    - MIXED
    - UNKNOWN

    Sınıflandırmada kullanılan temel sinyaller:

    - Native PDF metni
    - Metin bloklarının kapladığı alan
    - Sayısal ifade yoğunluğu
    - Tabloya benzeyen satırların oranı
    - PyMuPDF tablo tespiti
    - Görsel blokların sayısı ve kapladığı alan
    - Vektörel çizim yoğunluğu
    - Grafik ve şekil başlıkları
    - GRI, ESRS, SASB ve TCFD kodları
    - Standart indeksi anahtar sözcükleri

    Native metni bulunmayan ve sayfanın büyük bölümü görselden oluşan
    sayfalar OCR metni verilene kadar UNKNOWN olarak işaretlenir.
    """

    _WORD_RE = re.compile(
        r"\b[\wÀ-ž]+\b",
        re.UNICODE,
    )

    _NUMBER_RE = re.compile(
        r"(?<!\w)"
        r"[+-]?"
        r"(?:"
        r"\d{1,3}(?:[.,\s]\d{3})+"
        r"|"
        r"\d+"
        r")"
        r"(?:[.,]\d+)?"
        r"%?"
        r"(?!\w)"
    )

    _MULTI_SPACE_RE = re.compile(r"\s{2,}")

    _GRAPHIC_CAPTION_RE = re.compile(
        r"(?im)^"
        r"\s*"
        r"(?:"
        r"figure|"
        r"fig\.?|"
        r"chart|"
        r"graph|"
        r"grafik|"
        r"şekil|"
        r"infographic|"
        r"infografik"
        r")"
        r"\s*"
        r"\d*"
        r"[\s.:–—-]"
    )

    _STANDARD_KEYWORDS = (
        "gri content index",
        "gri index",
        "gri içerik endeksi",
        "gri içerik indeksi",
        "gri standartları içerik endeksi",
        "esrs content index",
        "esrs disclosure requirement",
        "esrs açıklama gerekliliği",
        "sasb content index",
        "sasb index",
        "sustainability accounting standards board",
        "standard index",
        "standards index",
        "standartlar indeksi",
        "standart içerik endeksi",
        "un global compact index",
        "ungc index",
        "tcfd index",
    )

    _STANDARD_CODE_PATTERNS = (
        re.compile(
            r"\bGRI\s*\d{3}(?:-\d{1,2})?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:2|[1-4]\d{2})-\d{1,2}\b",
        ),
        re.compile(
            r"\bESRS\s+[A-Z]\d(?:-\d+)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:E|S|G)\d-\d+\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:SASB|TCFD)\b",
            re.IGNORECASE,
        ),
    )

    def __init__(
        self,
        *,
        min_native_text_chars: int = 80,
        min_native_words: int = 12,
        scanned_image_ratio_threshold: float = 0.55,
        min_detected_score: float = 0.30,
        mixed_score_threshold: float = 0.45,
        mixed_score_margin: float = 0.16,
        enable_table_detection: bool = True,
    ) -> None:
        if min_native_text_chars < 0:
            raise ValueError(
                "min_native_text_chars negatif olamaz."
            )

        if min_native_words < 0:
            raise ValueError(
                "min_native_words negatif olamaz."
            )

        ratio_values = {
            "scanned_image_ratio_threshold": (
                scanned_image_ratio_threshold
            ),
            "min_detected_score": min_detected_score,
            "mixed_score_threshold": mixed_score_threshold,
            "mixed_score_margin": mixed_score_margin,
        }

        for name, value in ratio_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} 0 ile 1 arasında olmalıdır."
                )

        self.min_native_text_chars = min_native_text_chars
        self.min_native_words = min_native_words
        self.scanned_image_ratio_threshold = (
            scanned_image_ratio_threshold
        )
        self.min_detected_score = min_detected_score
        self.mixed_score_threshold = mixed_score_threshold
        self.mixed_score_margin = mixed_score_margin
        self.enable_table_detection = enable_table_detection

    def classify_many(
        self,
        pages: Sequence[Any],
        *,
        ocr_text_by_page: Mapping[int, str] | None = None,
    ) -> list[PageClassification]:
        """
        IngestionPipeline tarafından üretilen RenderedPage kayıtlarını
        toplu olarak sınıflandırır.

        PageRenderer her rapor klasörüne ``render_manifest.json`` yazar.
        Bu metot, RenderedPage içindeki ``image_path`` üzerinden manifesti
        bulur, kaynak PDF'i açar ve ilgili sayfayı sınıflandırır. Açık bir
        PyMuPDF Page nesnesi verilmesi de desteklenir.
        """
        page_items = list(pages)

        if not page_items:
            return []

        ocr_text_by_page = ocr_text_by_page or {}
        opened_documents: dict[Path, Any] = {}
        results: list[PageClassification] = []

        try:
            for item in page_items:
                if self._is_pymupdf_page(item):
                    page_number = int(item.number) + 1
                    results.append(
                        self.classify_page(
                            item,
                            ocr_text=ocr_text_by_page.get(page_number),
                        )
                    )
                    continue

                source_pdf = self._resolve_source_pdf(item)
                page_number = self._resolve_page_number(item)
                image_path = self._resolve_image_path(item)

                document = opened_documents.get(source_pdf)

                if document is None:
                    document = self._open_document(source_pdf)
                    opened_documents[source_pdf] = document

                if not 1 <= page_number <= document.page_count:
                    raise PageClassificationError(
                        f"Geçersiz sayfa numarası: {page_number}. "
                        f"PDF sayfa aralığı 1-{document.page_count}."
                    )

                results.append(
                    self.classify_page(
                        document.load_page(page_number - 1),
                        ocr_text=ocr_text_by_page.get(page_number),
                        source_pdf=source_pdf,
                        image_path=image_path,
                    )
                )

            logger.info(
                "%d render edilmiş sayfa sınıflandırıldı.",
                len(results),
            )

            return results
        finally:
            for document in opened_documents.values():
                document.close()

    def classify_pages(
        self,
        pages: Sequence[Any],
        *,
        ocr_text_by_page: Mapping[int, str] | None = None,
    ) -> list[PageClassification]:
        """``classify_many`` için uyumluluk takma adı."""
        return self.classify_many(
            pages,
            ocr_text_by_page=ocr_text_by_page,
        )

    def classify(
        self,
        page: Any,
        *,
        ocr_text: str | None = None,
    ) -> PageClassification:
        """
        Tek bir açık PyMuPDF sayfasını veya RenderedPage kaydını
        sınıflandırır. IngestionPipeline'ın tekil fallback çağrısıyla
        uyumludur.
        """
        if self._is_pymupdf_page(page):
            return self.classify_page(
                page,
                ocr_text=ocr_text,
            )

        source_pdf = self._resolve_source_pdf(page)
        page_number = self._resolve_page_number(page)
        image_path = self._resolve_image_path(page)
        document = self._open_document(source_pdf)

        try:
            if not 1 <= page_number <= document.page_count:
                raise PageClassificationError(
                    f"Geçersiz sayfa numarası: {page_number}. "
                    f"PDF sayfa aralığı 1-{document.page_count}."
                )

            return self.classify_page(
                document.load_page(page_number - 1),
                ocr_text=ocr_text,
                source_pdf=source_pdf,
                image_path=image_path,
            )
        finally:
            document.close()

    def classify_pdf(
        self,
        pdf_path: str | Path,
        *,
        page_numbers: Iterable[int] | None = None,
        ocr_text_by_page: Mapping[int, str] | None = None,
    ) -> list[PageClassification]:
        """
        PDF içindeki seçili veya bütün sayfaları sınıflandırır.

        Parametreler
        ------------
        pdf_path:
            Sınıflandırılacak PDF dosyasının yolu.

        page_numbers:
            Sınıflandırılacak 1 tabanlı sayfa numaraları.
            None verilirse bütün sayfalar sınıflandırılır.

        ocr_text_by_page:
            Daha önce OCR uygulanmış sayfaların metinleri.
            Sözlük anahtarları 1 tabanlı sayfa numaralarıdır.
        """
        pdf_path = Path(
            pdf_path
        ).expanduser().resolve()

        self._validate_pdf_path(pdf_path)

        ocr_text_by_page = ocr_text_by_page or {}

        try:
            document = pymupdf.open(pdf_path)
        except Exception as exc:
            raise PageClassificationError(
                f"PDF açılamadı: {pdf_path}"
            ) from exc

        try:
            if document.needs_pass:
                raise PageClassificationError(
                    "Parola korumalı PDF sınıflandırılamaz."
                )

            if document.page_count == 0:
                raise PageClassificationError(
                    "PDF içinde sayfa bulunamadı."
                )

            selected_pages = self._normalize_pages(
                page_numbers=page_numbers,
                page_count=document.page_count,
            )

            results: list[PageClassification] = []

            for page_number in selected_pages:
                page = document.load_page(
                    page_number - 1
                )

                result = self.classify_page(
                    page,
                    ocr_text=ocr_text_by_page.get(
                        page_number
                    ),
                    source_pdf=pdf_path,
                )

                results.append(result)

            logger.info(
                "%d sayfa sınıflandırıldı: %s",
                len(results),
                pdf_path,
            )

            return results
        finally:
            document.close()

    def classify_page(
        self,
        page: pymupdf.Page,
        *,
        ocr_text: str | None = None,
        source_pdf: str | Path | None = None,
        image_path: str | Path | None = None,
    ) -> PageClassification:
        """
        Açık bir PyMuPDF sayfasını sınıflandırır.

        OCR metni verilmişse ve native metin yetersizse sınıflandırmada
        OCR çıktısı kullanılır.
        """
        page_number = int(page.number) + 1

        try:
            native_text = page.get_text(
                "text",
                sort=True,
            ).strip()

            text_boxes = self._extract_text_boxes(
                page
            )

            image_boxes = self._extract_image_boxes(
                page
            )

            drawing_boxes = self._extract_drawing_boxes(
                page
            )

            table_boxes = self._extract_table_boxes(
                page
            )
        except Exception as exc:
            raise PageClassificationError(
                f"Sayfa {page_number} analiz edilemedi."
            ) from exc

        native_words = self._WORD_RE.findall(
            native_text
        )

        has_native_text = (
            len(native_text)
            >= self.min_native_text_chars
            and len(native_words)
            >= self.min_native_words
        )

        normalized_ocr_text = (
            ocr_text or ""
        ).strip()

        used_ocr_text = (
            not has_native_text
            and bool(normalized_ocr_text)
        )

        analysis_text = (
            normalized_ocr_text
            if used_ocr_text
            else native_text
        )

        page_area = max(
            float(
                page.rect.width
                * page.rect.height
            ),
            1.0,
        )

        image_area_ratio = self._calculate_area_ratio(
            boxes=image_boxes,
            page_rect=page.rect,
            page_area=page_area,
        )

        is_scanned = (
            not has_native_text
            and image_area_ratio
            >= self.scanned_image_ratio_threshold
        )

        needs_ocr = (
            is_scanned
            and not used_ocr_text
        )

        signals = self._build_signals(
            page=page,
            text=analysis_text,
            text_boxes=text_boxes,
            image_boxes=image_boxes,
            drawing_boxes=drawing_boxes,
            table_boxes=table_boxes,
        )

        scores = self._calculate_scores(
            signals
        )

        (
            primary_type,
            detected_types,
            confidence,
            reasons,
        ) = self._decide_classification(
            scores=scores,
            signals=signals,
            needs_ocr=needs_ocr,
            is_scanned=is_scanned,
            used_ocr_text=used_ocr_text,
        )

        return PageClassification(
            page_index=int(page.number),
            page_number=page_number,
            primary_type=primary_type,
            detected_types=detected_types,
            confidence=confidence,
            has_native_text=has_native_text,
            needs_ocr=needs_ocr,
            is_scanned=is_scanned,
            used_ocr_text=used_ocr_text,
            text=analysis_text,
            width_points=round(
                float(page.rect.width),
                3,
            ),
            height_points=round(
                float(page.rect.height),
                3,
            ),
            rotation=int(page.rotation),
            scores=scores,
            signals=signals,
            reasons=reasons,
            source_pdf=(
                Path(source_pdf).expanduser().resolve()
                if source_pdf is not None
                else None
            ),
            image_path=(
                Path(image_path).expanduser().resolve()
                if image_path is not None
                else None
            ),
        )

    @staticmethod
    def _is_pymupdf_page(value: Any) -> bool:
        """Nesnenin açık bir PyMuPDF sayfası olup olmadığını belirler."""
        return (
            hasattr(value, "number")
            and hasattr(value, "rect")
            and callable(getattr(value, "get_text", None))
            and callable(getattr(value, "get_drawings", None))
        )

    @staticmethod
    def _read_value(
        source: Any,
        *names: str,
    ) -> Any:
        for name in names:
            if isinstance(source, Mapping):
                value = source.get(name)
            else:
                value = getattr(source, name, None)

            if value is not None:
                return value

        return None

    @classmethod
    def _resolve_page_number(
        cls,
        page: Any,
    ) -> int:
        page_number = cls._read_value(
            page,
            "page_number",
            "number",
        )

        if page_number is None:
            page_index = cls._read_value(
                page,
                "page_index",
                "index",
            )

            if page_index is None:
                raise PageClassificationError(
                    "RenderedPage kaydında page_number veya page_index "
                    "bulunamadı."
                )

            page_number = int(page_index) + 1

        try:
            normalized_page_number = int(page_number)
        except (TypeError, ValueError) as exc:
            raise PageClassificationError(
                f"Geçersiz sayfa numarası: {page_number!r}"
            ) from exc

        if normalized_page_number <= 0:
            raise PageClassificationError(
                f"Sayfa numarası pozitif olmalıdır: "
                f"{normalized_page_number}"
            )

        return normalized_page_number

    @classmethod
    def _resolve_image_path(
        cls,
        page: Any,
    ) -> Path | None:
        value = cls._read_value(
            page,
            "image_path",
            "rendered_path",
            "path",
        )

        if value is None:
            return None

        return Path(value).expanduser().resolve()

    @classmethod
    def _resolve_source_pdf(
        cls,
        page: Any,
    ) -> Path:
        direct_value = cls._read_value(
            page,
            "source_pdf",
            "pdf_path",
            "source_path",
        )

        if direct_value is not None:
            source_pdf = Path(direct_value).expanduser().resolve()
            cls._validate_pdf_path(source_pdf)
            return source_pdf

        image_path = cls._resolve_image_path(page)

        if image_path is None:
            raise PageClassificationError(
                "RenderedPage kaydında kaynak PDF veya image_path "
                "bilgisi bulunamadı."
            )

        manifest_path = image_path.parent / "render_manifest.json"

        if not manifest_path.is_file():
            raise PageClassificationError(
                f"Render manifesti bulunamadı: {manifest_path}"
            )

        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise PageClassificationError(
                f"Render manifesti okunamadı: {manifest_path}"
            ) from exc

        source_value = manifest.get("source_pdf")

        if not source_value:
            raise PageClassificationError(
                f"Render manifestinde source_pdf bulunamadı: "
                f"{manifest_path}"
            )

        source_pdf = Path(source_value).expanduser()

        if not source_pdf.is_absolute():
            source_pdf = manifest_path.parent / source_pdf

        source_pdf = source_pdf.resolve()
        cls._validate_pdf_path(source_pdf)
        return source_pdf

    @staticmethod
    def _open_document(
        pdf_path: Path,
    ) -> Any:
        try:
            document = pymupdf.open(pdf_path)
        except Exception as exc:
            raise PageClassificationError(
                f"PDF açılamadı: {pdf_path}"
            ) from exc

        if document.needs_pass:
            document.close()
            raise PageClassificationError(
                "Parola korumalı PDF sınıflandırılamaz."
            )

        if document.page_count == 0:
            document.close()
            raise PageClassificationError(
                "PDF içinde sayfa bulunamadı."
            )

        return document

    def _build_signals(
        self,
        *,
        page: pymupdf.Page,
        text: str,
        text_boxes: Sequence[pymupdf.Rect],
        image_boxes: Sequence[pymupdf.Rect],
        drawing_boxes: Sequence[pymupdf.Rect],
        table_boxes: Sequence[pymupdf.Rect],
    ) -> dict[str, int | float]:
        page_area = max(
            float(
                page.rect.width
                * page.rect.height
            ),
            1.0,
        )

        words = self._WORD_RE.findall(
            text
        )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        number_count = len(
            self._NUMBER_RE.findall(text)
        )

        tabular_line_count = sum(
            self._looks_like_tabular_line(line)
            for line in lines
        )

        normalized_text = self._normalize_text(
            text
        )

        standard_keyword_hits = sum(
            normalized_text.count(keyword)
            for keyword
            in self._STANDARD_KEYWORDS
        )

        standard_code_hits = sum(
            len(pattern.findall(text))
            for pattern
            in self._STANDARD_CODE_PATTERNS
        )

        graphic_caption_hits = len(
            self._GRAPHIC_CAPTION_RE.findall(
                text
            )
        )

        return {
            "text_char_count": len(text),
            "word_count": len(words),
            "line_count": len(lines),
            "text_block_count": len(
                text_boxes
            ),
            "text_area_ratio": (
                self._calculate_area_ratio(
                    boxes=text_boxes,
                    page_rect=page.rect,
                    page_area=page_area,
                )
            ),
            "numeric_token_ratio": (
                self._safe_ratio(
                    number_count,
                    len(words),
                )
            ),
            "tabular_line_ratio": (
                self._safe_ratio(
                    tabular_line_count,
                    len(lines),
                )
            ),
            "image_count": len(
                image_boxes
            ),
            "image_area_ratio": (
                self._calculate_area_ratio(
                    boxes=image_boxes,
                    page_rect=page.rect,
                    page_area=page_area,
                )
            ),
            "drawing_count": len(
                drawing_boxes
            ),
            "drawing_area_ratio": (
                self._calculate_area_ratio(
                    boxes=drawing_boxes,
                    page_rect=page.rect,
                    page_area=page_area,
                )
            ),
            "table_count": len(
                table_boxes
            ),
            "table_area_ratio": (
                self._calculate_area_ratio(
                    boxes=table_boxes,
                    page_rect=page.rect,
                    page_area=page_area,
                )
            ),
            "standard_keyword_hits": (
                standard_keyword_hits
            ),
            "standard_code_hits": (
                standard_code_hits
            ),
            "graphic_caption_hits": (
                graphic_caption_hits
            ),
        }

    def _calculate_scores(
        self,
        signals: Mapping[str, int | float],
    ) -> dict[str, float]:
        values = {
            key: float(value)
            for key, value
            in signals.items()
        }

        text_score = (
            0.30
            * self._clamp(
                values["text_char_count"]
                / 2500
            )
            + 0.25
            * self._clamp(
                values["word_count"]
                / 350
            )
            + 0.20
            * self._clamp(
                values["text_block_count"]
                / 12
            )
            + 0.25
            * self._clamp(
                values["text_area_ratio"]
                / 0.55
            )
        )

        table_score = (
            0.34
            * self._clamp(
                values["table_count"]
                / 2
            )
            + 0.24
            * self._clamp(
                values["table_area_ratio"]
                / 0.45
            )
            + 0.18
            * self._clamp(
                values["tabular_line_ratio"]
                / 0.45
            )
            + 0.14
            * self._clamp(
                values["numeric_token_ratio"]
                / 0.30
            )
            + 0.10
            * self._clamp(
                values["drawing_count"]
                / 80
            )
        )

        graphic_score = (
            0.26
            * self._clamp(
                values["image_area_ratio"]
                / 0.60
            )
            + 0.12
            * self._clamp(
                values["image_count"]
                / 3
            )
            + 0.18
            * self._clamp(
                values["drawing_count"]
                / 12
            )
            + 0.14
            * self._clamp(
                values["drawing_area_ratio"]
                / 0.35
            )
            + 0.30
            * self._clamp(
                values["graphic_caption_hits"]
            )
        )

        standard_score = (
            0.46
            * self._clamp(
                values["standard_keyword_hits"]
                / 2
            )
            + 0.34
            * self._clamp(
                values["standard_code_hits"]
                / 8
            )
            + 0.15
            * table_score
            + 0.05
            * text_score
        )

        # Tablo veya grafik sinyali çok güçlüyse salt metin skorunun
        # gereksiz biçimde baskın hâle gelmesini engeller.
        text_score *= (
            1.0
            - 0.30
            * max(
                table_score,
                graphic_score,
            )
        )

        return {
            PageType.TEXT.value: (
                self._round_score(
                    text_score
                )
            ),
            PageType.TABLE.value: (
                self._round_score(
                    table_score
                )
            ),
            PageType.GRAPHIC.value: (
                self._round_score(
                    graphic_score
                )
            ),
            PageType.STANDARD_INDEX.value: (
                self._round_score(
                    standard_score
                )
            ),
        }

    def _decide_classification(
        self,
        *,
        scores: Mapping[str, float],
        signals: Mapping[str, int | float],
        needs_ocr: bool,
        is_scanned: bool,
        used_ocr_text: bool,
    ) -> tuple[
        PageType,
        tuple[PageType, ...],
        float,
        tuple[str, ...],
    ]:
        reasons: list[str] = []

        if needs_ocr:
            reasons.append(
                "Native metin yetersiz; "
                "içerik sınıfı için OCR gereklidir."
            )

            if is_scanned:
                reasons.append(
                    "Sayfanın büyük bölümü "
                    "görsel içerikten oluşuyor."
                )

            image_area_ratio = float(
                signals["image_area_ratio"]
            )

            confidence = self._round_score(
                0.55
                + 0.35
                * image_area_ratio
            )

            return (
                PageType.UNKNOWN,
                (PageType.UNKNOWN,),
                confidence,
                tuple(reasons),
            )

        ranked_types = sorted(
            (
                (
                    PageType.TEXT,
                    scores[
                        PageType.TEXT.value
                    ],
                ),
                (
                    PageType.TABLE,
                    scores[
                        PageType.TABLE.value
                    ],
                ),
                (
                    PageType.GRAPHIC,
                    scores[
                        PageType.GRAPHIC.value
                    ],
                ),
                (
                    PageType.STANDARD_INDEX,
                    scores[
                        PageType.STANDARD_INDEX.value
                    ],
                ),
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        top_type, top_score = (
            ranked_types[0]
        )

        second_type, second_score = (
            ranked_types[1]
        )

        detected_types = tuple(
            page_type
            for page_type, score
            in ranked_types
            if score
            >= self.min_detected_score
        )

        standard_is_explicit = (
            scores[
                PageType.STANDARD_INDEX.value
            ]
            >= 0.52
            and int(
                signals[
                    "standard_keyword_hits"
                ]
            )
            >= 1
            and int(
                signals[
                    "standard_code_hits"
                ]
            )
            >= 2
        )

        if standard_is_explicit:
            primary_type = (
                PageType.STANDARD_INDEX
            )

            detected_types = tuple(
                dict.fromkeys(
                    (
                        PageType.STANDARD_INDEX,
                        *detected_types,
                    )
                )
            )

            confidence = self._round_score(
                0.58
                + 0.40
                * scores[
                    PageType.STANDARD_INDEX.value
                ]
            )

            reasons.append(
                "Standart indeksi başlığı ve "
                "disclosure kodları birlikte bulundu."
            )

        elif top_score < self.min_detected_score:
            return (
                PageType.UNKNOWN,
                (PageType.UNKNOWN,),
                0.40,
                (
                    "Hiçbir içerik türü "
                    "yeterli skora ulaşmadı.",
                ),
            )

        elif (
            top_score
            >= self.mixed_score_threshold
            and second_score
            >= self.mixed_score_threshold
            and top_score - second_score
            <= self.mixed_score_margin
        ):
            primary_type = PageType.MIXED

            confidence = self._round_score(
                (
                    top_score
                    + second_score
                )
                / 2
            )

            reasons.append(
                f"{top_type.value} ve "
                f"{second_type.value} "
                "sinyalleri birlikte güçlü."
            )

        else:
            primary_type = top_type

            margin = max(
                top_score - second_score,
                0.0,
            )

            confidence = self._round_score(
                min(
                    0.99,
                    0.48
                    + 0.38
                    * top_score
                    + 0.28
                    * margin,
                )
            )

            reasons.append(
                "En güçlü sınıf skoru: "
                f"{top_type.value}="
                f"{top_score:.3f}."
            )

        table_count = int(
            signals["table_count"]
        )

        if table_count > 0:
            reasons.append(
                f"{table_count} tablo bölgesi "
                "tespit edildi."
            )

        if int(
            signals["graphic_caption_hits"]
        ) > 0:
            reasons.append(
                "Şekil veya grafik başlığı bulundu."
            )

        if float(
            signals["image_area_ratio"]
        ) >= 0.25:
            reasons.append(
                "Sayfada anlamlı büyüklükte "
                "görsel alan bulunuyor."
            )

        if used_ocr_text:
            reasons.append(
                "Sınıflandırmada OCR metni kullanıldı."
            )

        return (
            primary_type,
            detected_types,
            confidence,
            tuple(reasons[:6]),
        )

    def _extract_text_boxes(
        self,
        page: pymupdf.Page,
    ) -> list[pymupdf.Rect]:
        boxes: list[pymupdf.Rect] = []

        blocks = page.get_text(
            "blocks",
            sort=True,
        )

        for block in blocks:
            if len(block) < 7:
                continue

            block_type = int(
                block[6]
            )

            if block_type != 0:
                continue

            block_text = str(
                block[4]
            ).strip()

            if not block_text:
                continue

            rect = self._safe_rect(
                block[:4],
                page.rect,
            )

            if rect is not None:
                boxes.append(rect)

        return boxes

    def _extract_image_boxes(
        self,
        page: pymupdf.Page,
    ) -> list[pymupdf.Rect]:
        boxes: list[pymupdf.Rect] = []

        try:
            image_info = page.get_image_info(
                xrefs=False
            )
        except Exception:
            image_info = []

        for image in image_info:
            rect = self._safe_rect(
                image.get("bbox"),
                page.rect,
            )

            if rect is not None:
                boxes.append(rect)

        if boxes:
            return boxes

        # get_image_info sonuç vermeyen bazı PDF türleri için
        # görüntü blokları ikinci yol olarak kullanılır.
        blocks = page.get_text(
            "blocks",
            sort=False,
        )

        for block in blocks:
            if len(block) < 7:
                continue

            block_type = int(
                block[6]
            )

            if block_type != 1:
                continue

            rect = self._safe_rect(
                block[:4],
                page.rect,
            )

            if rect is not None:
                boxes.append(rect)

        return boxes

    def _extract_drawing_boxes(
        self,
        page: pymupdf.Page,
    ) -> list[pymupdf.Rect]:
        boxes: list[pymupdf.Rect] = []

        drawings = page.get_drawings()

        for drawing in drawings:
            rect = self._safe_rect(
                drawing.get("rect"),
                page.rect,
            )

            if rect is not None:
                boxes.append(rect)

        return boxes

    def _extract_table_boxes(
        self,
        page: pymupdf.Page,
    ) -> list[pymupdf.Rect]:
        if not self.enable_table_detection:
            return []

        if not hasattr(
            page,
            "find_tables",
        ):
            logger.warning(
                "Kullanılan PyMuPDF sürümünde "
                "Page.find_tables bulunmuyor."
            )

            return []

        try:
            finder = page.find_tables()
        except Exception as exc:
            logger.debug(
                "Sayfa %d tablo tespiti başarısız: %s",
                int(page.number) + 1,
                exc,
            )

            return []

        boxes: list[pymupdf.Rect] = []

        for table in getattr(
            finder,
            "tables",
            [],
        ):
            rect = self._safe_rect(
                getattr(
                    table,
                    "bbox",
                    None,
                ),
                page.rect,
            )

            if rect is not None:
                boxes.append(rect)

        return boxes

    @classmethod
    def _looks_like_tabular_line(
        cls,
        line: str,
    ) -> bool:
        words = cls._WORD_RE.findall(
            line
        )

        number_count = len(
            cls._NUMBER_RE.findall(
                line
            )
        )

        has_column_spacing = (
            "\t" in line
            or bool(
                cls._MULTI_SPACE_RE.search(
                    line
                )
            )
        )

        return (
            len(words) >= 3
            and (
                number_count >= 2
                or (
                    number_count >= 1
                    and has_column_spacing
                )
                or (
                    has_column_spacing
                    and len(words) >= 5
                )
            )
        )

    @staticmethod
    def _safe_rect(
        value: Any,
        page_rect: pymupdf.Rect,
    ) -> pymupdf.Rect | None:
        if value is None:
            return None

        try:
            rect = (
                pymupdf.Rect(value)
                & page_rect
            )
        except Exception:
            return None

        if rect.is_empty:
            return None

        if rect.is_infinite:
            return None

        if rect.width <= 0:
            return None

        if rect.height <= 0:
            return None

        return rect

    @staticmethod
    def _calculate_area_ratio(
        *,
        boxes: Sequence[pymupdf.Rect],
        page_rect: pymupdf.Rect,
        page_area: float,
    ) -> float:
        total_area = 0.0

        for box in boxes:
            clipped_box = (
                box
                & page_rect
            )

            if clipped_box.is_empty:
                continue

            if clipped_box.is_infinite:
                continue

            total_area += max(
                float(
                    clipped_box.width
                    * clipped_box.height
                ),
                0.0,
            )

        return round(
            min(
                total_area
                / max(page_area, 1.0),
                1.0,
            ),
            4,
        )

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        translation_table = str.maketrans(
            {
                "İ": "i",
                "I": "i",
                "ı": "i",
                "Ş": "ş",
                "Ğ": "ğ",
                "Ü": "ü",
                "Ö": "ö",
                "Ç": "ç",
            }
        )

        translated = text.translate(
            translation_table
        )

        return " ".join(
            translated.lower().split()
        )

    @staticmethod
    def _safe_ratio(
        numerator: int | float,
        denominator: int | float,
    ) -> float:
        if denominator <= 0:
            return 0.0

        return round(
            float(numerator)
            / float(denominator),
            4,
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        if not math.isfinite(value):
            return 0.0

        return min(
            max(value, 0.0),
            1.0,
        )

    @classmethod
    def _round_score(
        cls,
        value: float,
    ) -> float:
        return round(
            cls._clamp(value),
            4,
        )

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
                int(number)
                for number
                in page_numbers
            )
        )

        if not pages:
            raise ValueError(
                "En az bir sayfa numarası verilmelidir."
            )

        invalid_pages = [
            number
            for number
            in pages
            if not 1 <= number <= page_count
        ]

        if invalid_pages:
            raise ValueError(
                f"Geçersiz sayfalar: {invalid_pages}. "
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
                "Yalnızca .pdf uzantılı "
                "dosyalar desteklenir."
            )


def classify_pdf_pages(
    pdf_path: str | Path,
    *,
    page_numbers: Iterable[int] | None = None,
    ocr_text_by_page: Mapping[int, str] | None = None,
    enable_table_detection: bool = True,
) -> list[PageClassification]:
    """PageClassifier için kısa kullanım arayüzü."""

    classifier = PageClassifier(
        enable_table_detection=(
            enable_table_detection
        )
    )

    return classifier.classify_pdf(
        pdf_path=pdf_path,
        page_numbers=page_numbers,
        ocr_text_by_page=ocr_text_by_page,
    )


__all__ = [
    "PageClassification",
    "PageClassificationError",
    "PageClassifier",
    "classify_pdf_pages",
]
