from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageFilter, ImageOps

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - OpenCV isteğe bağlıdır.
    cv2 = None
    np = None


@dataclass(slots=True, frozen=True)
class PageChartBoundingBox:
    """Render edilmiş sayfa görüntüsü üzerindeki piksel koordinatları."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("Grafik bounding box koordinatları geçersiz.")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def clamp(
        self,
        *,
        width: int,
        height: int,
    ) -> "PageChartBoundingBox":
        x0 = max(min(self.x0, width - 1), 0)
        y0 = max(min(self.y0, height - 1), 0)
        x1 = max(min(self.x1, width), x0 + 1)
        y1 = max(min(self.y1, height), y0 + 1)
        return PageChartBoundingBox(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
        )


@dataclass(slots=True, frozen=True)
class PageChartCandidate:
    page_number: int
    source_image_path: Path
    bounding_box: PageChartBoundingBox
    confidence: float

    detection_method: str
    chart_signals: tuple[str, ...]

    width_ratio: float
    height_ratio: float
    area_ratio: float

    nearby_text: str | None = None
    title_hint: str | None = None
    crop_path: Path | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def candidate_id(self) -> str:
        payload = (
            f"{self.page_number}:"
            f"{self.bounding_box.x0}:"
            f"{self.bounding_box.y0}:"
            f"{self.bounding_box.x1}:"
            f"{self.bounding_box.y1}:"
            f"{self.source_image_path}"
        )
        digest = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()[:16]
        return f"page_{self.page_number:04d}_chart_{digest}"


@dataclass(slots=True, frozen=True)
class PageChartDetectionResult:
    page_number: int
    source_image_path: Path
    candidates: tuple[PageChartCandidate, ...]
    warnings: tuple[str, ...] = ()

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


class PageChartDetector:
    """
    Render edilmiş tam sayfa görüntülerinde grafik bölgesi adaylarını tespit eder.

    Tasarım:
    1. Yerel geometrik sinyallerle aday bölgeler üretir.
    2. Yıl/sayı yoğunluğu ve grafik bağlamı ile adayları puanlar.
    3. Logo, framework ve dekoratif sayfaları baskılar.
    4. Yalnız güçlü adayları Groq Vision'a gönderilecek biçimde döndürür.

    OpenCV varsa çizgi, kontur ve tekrar eden bar geometrileri kullanılır.
    OpenCV yoksa PIL tabanlı daha hafif bir fallback uygulanır.
    """

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.70,
        minimum_area_ratio: float = 0.035,
        maximum_area_ratio: float = 0.78,
        minimum_width_ratio: float = 0.28,
        minimum_height_ratio: float = 0.12,
        maximum_candidates_per_page: int = 3,
        crop_padding_ratio: float = 0.025,
        save_crops: bool = True,
    ) -> None:
        self.minimum_confidence = self._clamp01(
            minimum_confidence
        )
        self.minimum_area_ratio = self._clamp01(
            minimum_area_ratio
        )
        self.maximum_area_ratio = self._clamp01(
            maximum_area_ratio
        )
        self.minimum_width_ratio = self._clamp01(
            minimum_width_ratio
        )
        self.minimum_height_ratio = self._clamp01(
            minimum_height_ratio
        )
        self.maximum_candidates_per_page = max(
            int(maximum_candidates_per_page),
            1,
        )
        self.crop_padding_ratio = max(
            float(crop_padding_ratio),
            0.0,
        )
        self.save_crops = bool(save_crops)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_pages(
        self,
        pages: Sequence[Any],
        *,
        output_directory: str | Path | None = None,
    ) -> list[PageChartDetectionResult]:
        results: list[PageChartDetectionResult] = []

        for index, page in enumerate(pages):
            page_number = int(
                getattr(page, "page_number", index + 1)
                or index + 1
            )
            image_path = self._extract_image_path(page)

            if image_path is None:
                results.append(
                    PageChartDetectionResult(
                        page_number=page_number,
                        source_image_path=Path(),
                        candidates=(),
                        warnings=(
                            "Sayfa için render görüntüsü bulunamadı.",
                        ),
                    )
                )
                continue

            page_text = self._extract_page_text(page)
            results.append(
                self.detect_page(
                    image_path,
                    page_number=page_number,
                    page_text=page_text,
                    output_directory=output_directory,
                )
            )

        return results

    def detect_page(
        self,
        image_path: str | Path,
        *,
        page_number: int,
        page_text: str | None = None,
        output_directory: str | Path | None = None,
    ) -> PageChartDetectionResult:
        source_path = Path(
            image_path
        ).expanduser().resolve()

        warnings: list[str] = []

        if not source_path.is_file():
            return PageChartDetectionResult(
                page_number=page_number,
                source_image_path=source_path,
                candidates=(),
                warnings=(
                    f"Sayfa görüntüsü bulunamadı: {source_path}",
                ),
            )

        try:
            with Image.open(source_path) as image:
                rgb_image = image.convert("RGB")
                width, height = rgb_image.size

                normalized_page_text = self._normalize_text(
                    page_text or ""
                )

                if self._is_framework_or_logo_page(
                    normalized_page_text
                ):
                    return PageChartDetectionResult(
                        page_number=page_number,
                        source_image_path=source_path,
                        candidates=(),
                        warnings=(),
                    )

                if cv2 is not None and np is not None:
                    raw_candidates = (
                        self._detect_with_opencv(
                            rgb_image,
                            normalized_page_text,
                        )
                    )
                    method = "opencv_geometry"
                else:
                    raw_candidates = (
                        self._detect_with_pil(
                            rgb_image,
                            normalized_page_text,
                        )
                    )
                    method = "pil_edge_density"
                    warnings.append(
                        "OpenCV bulunamadığı için PIL tabanlı "
                        "grafik tespiti kullanıldı."
                    )

                candidates: list[PageChartCandidate] = []

                for raw_box, raw_score, signals in raw_candidates:
                    padded_box = self._pad_box(
                        raw_box,
                        image_width=width,
                        image_height=height,
                    )
                    score = self._clamp01(raw_score)

                    width_ratio = (
                        padded_box.width / max(width, 1)
                    )
                    height_ratio = (
                        padded_box.height / max(height, 1)
                    )
                    area_ratio = (
                        padded_box.area
                        / max(width * height, 1)
                    )

                    if not self._passes_geometry_filter(
                        width_ratio=width_ratio,
                        height_ratio=height_ratio,
                        area_ratio=area_ratio,
                    ):
                        continue

                    if score < self.minimum_confidence:
                        continue

                    crop_path = None
                    if self.save_crops:
                        try:
                            crop_path = self._save_crop(
                                image=rgb_image,
                                source_image_path=source_path,
                                page_number=page_number,
                                bounding_box=padded_box,
                                output_directory=output_directory,
                            )
                        except Exception as exc:
                            warnings.append(
                                "Grafik adayı kırpılamadı "
                                f"(sayfa {page_number}): {exc}"
                            )

                    candidates.append(
                        PageChartCandidate(
                            page_number=page_number,
                            source_image_path=source_path,
                            bounding_box=padded_box,
                            confidence=round(score, 4),
                            detection_method=method,
                            chart_signals=tuple(
                                dict.fromkeys(signals)
                            ),
                            width_ratio=round(
                                width_ratio,
                                5,
                            ),
                            height_ratio=round(
                                height_ratio,
                                5,
                            ),
                            area_ratio=round(
                                area_ratio,
                                5,
                            ),
                            nearby_text=page_text,
                            title_hint=(
                                self._extract_title_hint(
                                    page_text or ""
                                )
                            ),
                            crop_path=crop_path,
                            metadata={
                                "image_width": width,
                                "image_height": height,
                                "opencv_available": (
                                    cv2 is not None
                                    and np is not None
                                ),
                                "detector_version": "page_chart_v1",
                            },
                        )
                    )

                candidates = self._deduplicate_candidates(
                    candidates
                )
                candidates = sorted(
                    candidates,
                    key=lambda item: (
                        -item.confidence,
                        -item.area_ratio,
                    ),
                )[: self.maximum_candidates_per_page]

                return PageChartDetectionResult(
                    page_number=page_number,
                    source_image_path=source_path,
                    candidates=tuple(candidates),
                    warnings=tuple(warnings),
                )

        except Exception as exc:
            return PageChartDetectionResult(
                page_number=page_number,
                source_image_path=source_path,
                candidates=(),
                warnings=(
                    "Sayfa grafik tespiti başarısız: "
                    f"{type(exc).__name__}: {exc}",
                ),
            )

    # ------------------------------------------------------------------
    # OpenCV detector
    # ------------------------------------------------------------------
    def _detect_with_opencv(
        self,
        image: Image.Image,
        page_text: str,
    ) -> list[
        tuple[PageChartBoundingBox, float, list[str]]
    ]:
        array = np.array(image)
        gray = cv2.cvtColor(
            array,
            cv2.COLOR_RGB2GRAY,
        )

        # Metin ve ince grafik çizgilerini korumak için hafif blur.
        blurred = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10,
        )

        image_height, image_width = gray.shape

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(int(image_width * 0.035), 18),
                1,
            ),
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                max(int(image_height * 0.025), 18),
            ),
        )

        horizontal = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            horizontal_kernel,
        )
        vertical = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            vertical_kernel,
        )

        line_mask = cv2.bitwise_or(
            horizontal,
            vertical,
        )

        contour_source = cv2.dilate(
            cv2.bitwise_or(binary, line_mask),
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (9, 9),
            ),
            iterations=2,
        )

        contours, _ = cv2.findContours(
            contour_source,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        page_signal_score, page_signals = (
            self._page_text_chart_score(page_text)
        )

        raw: list[
            tuple[PageChartBoundingBox, float, list[str]]
        ] = []

        for contour in contours:
            x, y, width, height = cv2.boundingRect(
                contour
            )
            if width <= 0 or height <= 0:
                continue

            box = PageChartBoundingBox(
                x0=int(x),
                y0=int(y),
                x1=int(x + width),
                y1=int(y + height),
            )

            width_ratio = width / max(
                image_width,
                1,
            )
            height_ratio = height / max(
                image_height,
                1,
            )
            area_ratio = box.area / max(
                image_width * image_height,
                1,
            )

            if not self._passes_geometry_filter(
                width_ratio=width_ratio,
                height_ratio=height_ratio,
                area_ratio=area_ratio,
            ):
                continue

            region_binary = binary[
                y:y + height,
                x:x + width,
            ]
            region_horizontal = horizontal[
                y:y + height,
                x:x + width,
            ]
            region_vertical = vertical[
                y:y + height,
                x:x + width,
            ]

            ink_density = float(
                np.count_nonzero(region_binary)
            ) / max(region_binary.size, 1)

            horizontal_density = float(
                np.count_nonzero(region_horizontal)
            ) / max(region_horizontal.size, 1)

            vertical_density = float(
                np.count_nonzero(region_vertical)
            ) / max(region_vertical.size, 1)

            repeated_bar_score = (
                self._repeated_bar_score(
                    region_binary
                )
            )
            color_diversity = (
                self._color_diversity_score(
                    array[
                        y:y + height,
                        x:x + width,
                    ]
                )
            )

            score = 0.0
            signals: list[str] = []

            if horizontal_density >= 0.012:
                score += 0.18
                signals.append(
                    "horizontal_lines"
                )

            if vertical_density >= 0.009:
                score += 0.18
                signals.append(
                    "vertical_lines"
                )

            if (
                horizontal_density >= 0.010
                and vertical_density >= 0.007
            ):
                score += 0.14
                signals.append(
                    "axis_like_geometry"
                )

            if repeated_bar_score >= 0.35:
                score += min(
                    repeated_bar_score * 0.28,
                    0.24,
                )
                signals.append(
                    "repeated_bar_geometry"
                )

            if color_diversity >= 0.30:
                score += min(
                    color_diversity * 0.18,
                    0.14,
                )
                signals.append(
                    "series_color_diversity"
                )

            if 0.035 <= ink_density <= 0.42:
                score += 0.08
                signals.append(
                    "chart_like_ink_density"
                )

            if width_ratio >= 0.45:
                score += 0.08
                signals.append(
                    "wide_visual_region"
                )

            if 0.14 <= height_ratio <= 0.62:
                score += 0.06
                signals.append(
                    "chart_like_aspect_ratio"
                )

            score += page_signal_score
            signals.extend(page_signals)

            # Yalnız renkli/dekoratif ama eksensiz görsellerin skoru baskılanır.
            has_geometry = (
                horizontal_density >= 0.010
                or vertical_density >= 0.008
                or repeated_bar_score >= 0.35
            )
            if not has_geometry:
                score = min(score, 0.56)

            raw.append(
                (
                    box,
                    self._clamp01(score),
                    signals,
                )
            )

        return self._merge_nearby_raw_candidates(
            raw,
            image_width=image_width,
            image_height=image_height,
        )

    # ------------------------------------------------------------------
    # PIL fallback detector
    # ------------------------------------------------------------------
    def _detect_with_pil(
        self,
        image: Image.Image,
        page_text: str,
    ) -> list[
        tuple[PageChartBoundingBox, float, list[str]]
    ]:
        gray = ImageOps.grayscale(image)
        edges = gray.filter(
            ImageFilter.FIND_EDGES
        )

        width, height = image.size
        tiles_x = 4
        tiles_y = 6
        tile_width = max(width // tiles_x, 1)
        tile_height = max(height // tiles_y, 1)

        page_signal_score, page_signals = (
            self._page_text_chart_score(page_text)
        )

        candidates: list[
            tuple[PageChartBoundingBox, float, list[str]]
        ] = []

        for row in range(tiles_y):
            for column in range(tiles_x):
                x0 = column * tile_width
                y0 = row * tile_height
                x1 = (
                    width
                    if column == tiles_x - 1
                    else (column + 1) * tile_width
                )
                y1 = (
                    height
                    if row == tiles_y - 1
                    else (row + 1) * tile_height
                )

                tile = edges.crop(
                    (x0, y0, x1, y1)
                )
                histogram = tile.histogram()
                edge_pixels = sum(
                    histogram[150:]
                )
                density = edge_pixels / max(
                    tile.width * tile.height,
                    1,
                )

                if density < 0.06:
                    continue

                score = min(
                    0.34 + density * 1.8,
                    0.62,
                )
                score += page_signal_score

                candidates.append(
                    (
                        PageChartBoundingBox(
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                        ),
                        self._clamp01(score),
                        [
                            "pil_edge_density",
                            *page_signals,
                        ],
                    )
                )

        return self._merge_nearby_raw_candidates(
            candidates,
            image_width=width,
            image_height=height,
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def _page_text_chart_score(
        self,
        normalized_text: str,
    ) -> tuple[float, list[str]]:
        explicit_terms = (
            "chart",
            "graph",
            "bar chart",
            "line chart",
            "pie chart",
            "grafik",
            "yillara gore",
            "trend",
            "distribution",
            "breakdown",
        )
        axis_terms = (
            "axis",
            "legend",
            "series",
            "eksen",
            "lejand",
            "seri",
        )

        years = set(
            re.findall(
                r"\b(?:19|20)\d{2}\b",
                normalized_text,
            )
        )
        numbers = re.findall(
            r"(?<!\w)[+-]?(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)"
            r"(?:[.,]\d+)?(?:\s*%)?(?!\w)",
            normalized_text,
        )

        score = 0.0
        signals: list[str] = []

        if any(
            term in normalized_text
            for term in explicit_terms
        ):
            score += 0.20
            signals.append(
                "explicit_chart_language"
            )

        if any(
            term in normalized_text
            for term in axis_terms
        ):
            score += 0.10
            signals.append(
                "axis_or_series_language"
            )

        if len(years) >= 3 and len(numbers) >= 6:
            score += 0.18
            signals.append(
                "multi_year_numeric_sequence"
            )
        elif len(years) >= 2 and len(numbers) >= 5:
            score += 0.10
            signals.append(
                "year_numeric_sequence"
            )

        return min(score, 0.34), signals

    @staticmethod
    def _repeated_bar_score(
        binary_region: Any,
    ) -> float:
        if (
            cv2 is None
            or np is None
            or binary_region.size == 0
        ):
            return 0.0

        contours, _ = cv2.findContours(
            binary_region,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        rectangles: list[
            tuple[int, int, int, int]
        ] = []

        region_height, region_width = (
            binary_region.shape
        )

        for contour in contours:
            x, y, width, height = (
                cv2.boundingRect(contour)
            )
            area = width * height

            if area < max(
                region_width * region_height * 0.001,
                25,
            ):
                continue

            aspect_ratio = max(
                width / max(height, 1),
                height / max(width, 1),
            )

            if aspect_ratio >= 1.7:
                rectangles.append(
                    (x, y, width, height)
                )

        if len(rectangles) < 3:
            return 0.0

        widths = [
            item[2]
            for item in rectangles
        ]
        heights = [
            item[3]
            for item in rectangles
        ]

        width_cv = (
            PageChartDetector._coefficient_of_variation(
                widths
            )
        )
        height_cv = (
            PageChartDetector._coefficient_of_variation(
                heights
            )
        )

        similarity = max(
            1.0 - min(width_cv, height_cv),
            0.0,
        )
        count_score = min(
            len(rectangles) / 10.0,
            1.0,
        )

        return (
            similarity * 0.65
            + count_score * 0.35
        )

    @staticmethod
    def _color_diversity_score(
        rgb_region: Any,
    ) -> float:
        if (
            np is None
            or rgb_region.size == 0
        ):
            return 0.0

        small = cv2.resize(
            rgb_region,
            (64, 64),
            interpolation=cv2.INTER_AREA,
        )
        pixels = small.reshape(
            -1,
            3,
        )

        # Beyaza çok yakın arka plan piksellerini dışla.
        visible = pixels[
            np.min(pixels, axis=1) < 235
        ]
        if visible.size == 0:
            return 0.0

        quantized = (
            visible // 32
        ).astype(np.uint8)
        unique_colors = np.unique(
            quantized,
            axis=0,
        )

        return min(
            len(unique_colors) / 18.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Candidate post-processing
    # ------------------------------------------------------------------
    def _merge_nearby_raw_candidates(
        self,
        candidates: Sequence[
            tuple[
                PageChartBoundingBox,
                float,
                list[str],
            ]
        ],
        *,
        image_width: int,
        image_height: int,
    ) -> list[
        tuple[
            PageChartBoundingBox,
            float,
            list[str],
        ]
    ]:
        ordered = sorted(
            candidates,
            key=lambda item: item[1],
            reverse=True,
        )
        merged: list[
            tuple[
                PageChartBoundingBox,
                float,
                list[str],
            ]
        ] = []

        for box, score, signals in ordered:
            merged_into_existing = False

            for index, (
                existing_box,
                existing_score,
                existing_signals,
            ) in enumerate(merged):
                iou = self._iou(
                    box,
                    existing_box,
                )
                close = self._boxes_are_close(
                    box,
                    existing_box,
                    image_width=image_width,
                    image_height=image_height,
                )

                if iou < 0.18 and not close:
                    continue

                union_box = PageChartBoundingBox(
                    x0=min(
                        box.x0,
                        existing_box.x0,
                    ),
                    y0=min(
                        box.y0,
                        existing_box.y0,
                    ),
                    x1=max(
                        box.x1,
                        existing_box.x1,
                    ),
                    y1=max(
                        box.y1,
                        existing_box.y1,
                    ),
                ).clamp(
                    width=image_width,
                    height=image_height,
                )

                merged[index] = (
                    union_box,
                    max(score, existing_score),
                    list(
                        dict.fromkeys(
                            [
                                *existing_signals,
                                *signals,
                            ]
                        )
                    ),
                )
                merged_into_existing = True
                break

            if not merged_into_existing:
                merged.append(
                    (box, score, list(signals))
                )

        return merged

    def _deduplicate_candidates(
        self,
        candidates: Sequence[
            PageChartCandidate
        ],
    ) -> list[PageChartCandidate]:
        ordered = sorted(
            candidates,
            key=lambda item: item.confidence,
            reverse=True,
        )
        result: list[PageChartCandidate] = []

        for candidate in ordered:
            if any(
                self._iou(
                    candidate.bounding_box,
                    existing.bounding_box,
                ) >= 0.72
                for existing in result
            ):
                continue
            result.append(candidate)

        return result

    def _passes_geometry_filter(
        self,
        *,
        width_ratio: float,
        height_ratio: float,
        area_ratio: float,
    ) -> bool:
        return (
            width_ratio >= self.minimum_width_ratio
            and height_ratio
            >= self.minimum_height_ratio
            and area_ratio
            >= self.minimum_area_ratio
            and area_ratio
            <= self.maximum_area_ratio
        )

    def _pad_box(
        self,
        box: PageChartBoundingBox,
        *,
        image_width: int,
        image_height: int,
    ) -> PageChartBoundingBox:
        padding_x = int(
            round(
                box.width
                * self.crop_padding_ratio
            )
        )
        padding_y = int(
            round(
                box.height
                * self.crop_padding_ratio
            )
        )

        return PageChartBoundingBox(
            x0=box.x0 - padding_x,
            y0=box.y0 - padding_y,
            x1=box.x1 + padding_x,
            y1=box.y1 + padding_y,
        ).clamp(
            width=image_width,
            height=image_height,
        )

    def _save_crop(
        self,
        *,
        image: Image.Image,
        source_image_path: Path,
        page_number: int,
        bounding_box: PageChartBoundingBox,
        output_directory: str | Path | None,
    ) -> Path:
        if output_directory is None:
            output_root = (
                source_image_path.parent
                / "page_chart_candidates"
            )
        else:
            output_root = Path(
                output_directory
            ).expanduser().resolve()

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        crop_hash = hashlib.sha256(
            (
                f"{page_number}:"
                f"{bounding_box.x0}:"
                f"{bounding_box.y0}:"
                f"{bounding_box.x1}:"
                f"{bounding_box.y1}"
            ).encode("utf-8")
        ).hexdigest()[:12]

        target_path = output_root / (
            f"page_{page_number:04d}_"
            f"chart_{crop_hash}.png"
        )

        image.crop(
            (
                bounding_box.x0,
                bounding_box.y0,
                bounding_box.x1,
                bounding_box.y1,
            )
        ).save(
            target_path,
            format="PNG",
            optimize=True,
        )

        return target_path

    # ------------------------------------------------------------------
    # General helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_image_path(
        page: Any,
    ) -> Path | None:
        for attribute in (
            "image_path",
            "rendered_image_path",
            "page_image_path",
        ):
            value = getattr(
                page,
                attribute,
                None,
            )
            if value:
                return Path(
                    value
                ).expanduser().resolve()

        metadata = getattr(
            page,
            "metadata",
            None,
        )
        if isinstance(metadata, Mapping):
            for key in (
                "image_path",
                "rendered_image_path",
                "page_image_path",
            ):
                value = metadata.get(key)
                if value:
                    return Path(
                        value
                    ).expanduser().resolve()

        return None

    @staticmethod
    def _extract_page_text(
        page: Any,
    ) -> str:
        parts: list[str] = []

        for attribute in (
            "text",
            "normalized_text",
        ):
            value = getattr(
                page,
                attribute,
                None,
            )
            if value:
                parts.append(
                    str(value)
                )

        for element in getattr(
            page,
            "elements",
            (),
        ) or ():
            value = getattr(
                element,
                "text",
                None,
            )
            if value:
                parts.append(
                    str(value)
                )

        return "\n".join(
            dict.fromkeys(
                item.strip()
                for item in parts
                if item.strip()
            )
        )

    @staticmethod
    def _extract_title_hint(
        text: str,
    ) -> str | None:
        for line in str(text or "").splitlines():
            cleaned = re.sub(
                r"\s+",
                " ",
                line,
            ).strip()
            if (
                5 <= len(cleaned) <= 180
                and not re.fullmatch(
                    r"[\d\s.,:%+-]+",
                    cleaned,
                )
            ):
                return cleaned
        return None

    @staticmethod
    def _is_framework_or_logo_page(
        normalized_text: str,
    ) -> bool:
        strong_phrases = (
            "reporting standards and frameworks",
            "standards and frameworks",
            "frameworks we comply with",
            "raporlama standartlari ve cerceveleri",
            "uyumlu oldugumuz raporlama standartlari",
        )
        tokens = (
            "ungc",
            "sdgs",
            "weps",
            "iirc",
            "sasb",
            "gri",
            "tcfd",
            "tsrs",
        )

        if any(
            phrase in normalized_text
            for phrase in strong_phrases
        ):
            return True

        token_hits = sum(
            1
            for token in tokens
            if re.search(
                rf"\b{re.escape(token)}\b",
                normalized_text,
            )
        )
        return token_hits >= 3

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        normalized = unicodedata.normalize(
            "NFKD",
            str(value or "").strip().casefold(),
        )
        ascii_text = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        ).replace(
            "ı",
            "i",
        )

        return re.sub(
            r"[^a-z0-9%²³.,:+-]+",
            " ",
            ascii_text,
        ).strip()

    @staticmethod
    def _boxes_are_close(
        first: PageChartBoundingBox,
        second: PageChartBoundingBox,
        *,
        image_width: int,
        image_height: int,
    ) -> bool:
        horizontal_gap = max(
            second.x0 - first.x1,
            first.x0 - second.x1,
            0,
        )
        vertical_gap = max(
            second.y0 - first.y1,
            first.y0 - second.y1,
            0,
        )

        return (
            horizontal_gap
            <= image_width * 0.025
            and vertical_gap
            <= image_height * 0.025
        )

    @staticmethod
    def _iou(
        first: PageChartBoundingBox,
        second: PageChartBoundingBox,
    ) -> float:
        x0 = max(
            first.x0,
            second.x0,
        )
        y0 = max(
            first.y0,
            second.y0,
        )
        x1 = min(
            first.x1,
            second.x1,
        )
        y1 = min(
            first.y1,
            second.y1,
        )

        if x1 <= x0 or y1 <= y0:
            return 0.0

        intersection = (
            (x1 - x0)
            * (y1 - y0)
        )
        union = (
            first.area
            + second.area
            - intersection
        )

        return (
            intersection / union
            if union > 0
            else 0.0
        )

    @staticmethod
    def _coefficient_of_variation(
        values: Iterable[int | float],
    ) -> float:
        numeric_values = [
            float(value)
            for value in values
        ]
        if not numeric_values:
            return 1.0

        mean = sum(
            numeric_values
        ) / len(numeric_values)

        if mean == 0:
            return 1.0

        variance = sum(
            (value - mean) ** 2
            for value in numeric_values
        ) / len(numeric_values)

        return math.sqrt(
            variance
        ) / abs(mean)

    @staticmethod
    def _clamp01(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(float(value), 1.0),
        )


__all__ = [
    "PageChartBoundingBox",
    "PageChartCandidate",
    "PageChartDetectionResult",
    "PageChartDetector",
]