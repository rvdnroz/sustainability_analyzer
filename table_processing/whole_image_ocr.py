from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import Output
except ImportError as exc:
    raise ImportError(
        "whole_image_ocr.py için pytesseract paketi gereklidir. "
        "'python -m pip install pytesseract' komutunu çalıştırın."
    ) from exc

from models.page import BoundingBox
from models.visual_evidence import (
    OCRLine,
    OCRWord,
    VisualAsset,
    WholeImageOCRResult,
)


logger = logging.getLogger(__name__)


class WholeImageOCRError(RuntimeError):
    """Bütün-görüntü OCR işlemi tamamlanamadığında yükseltilir."""


@dataclass(slots=True, frozen=True)
class WholeImageOCRConfig:
    """
    Bütün-görüntü OCR davranışını belirleyen ayarlar.

    `minimum_word_confidence` Tesseract'ın 0-100 aralığındaki güven
    değeridir. Sonuç modellerine yazılan güven değerleri ise 0-1
    aralığına normalize edilir.
    """

    language: str = "tur+eng"
    psm_modes: tuple[int, ...] = (6, 11)
    oem: int = 3

    minimum_word_confidence: float = 20.0
    minimum_result_confidence: float = 0.45
    minimum_text_characters: int = 6
    minimum_word_count: int = 2

    upscale_factor: float = 1.5
    maximum_image_dimension: int = 3000

    apply_deskew: bool = False
    maximum_deskew_angle: float = 12.0

    apply_denoising: bool = False
    apply_sharpening: bool = True

    preprocessing_variants: tuple[str, ...] = (
        "grayscale",
        "otsu",
    )

    line_vertical_tolerance_ratio: float = 0.70
    minimum_line_vertical_tolerance: float = 8.0

    column_gap_height_ratio: float = 1.50
    column_gap_character_ratio: float = 3.50
    minimum_column_gap_pixels: float = 18.0

    structured_minimum_rows: int = 2
    structured_minimum_columns: int = 2
    structured_row_ratio: float = 0.50

    # Temiz bir OCR sonucu erken bulunursa kalan kombinasyonları çalıştırma.
    enable_early_stop: bool = True
    early_stop_minimum_confidence: float = 0.86
    early_stop_minimum_score: float = 0.72
    early_stop_minimum_characters: int = 40
    early_stop_require_structured: bool = False

    keep_debug_images: bool = False
    debug_directory: Path | None = None

    tesseract_command: str | Path | None = None

    def __post_init__(self) -> None:
        if not self.language.strip():
            raise ValueError("language boş olamaz.")

        if not self.psm_modes:
            raise ValueError("En az bir Tesseract PSM modu tanımlanmalıdır.")

        invalid_psm = [
            value
            for value in self.psm_modes
            if not 0 <= int(value) <= 13
        ]
        if invalid_psm:
            raise ValueError(
                "PSM değerleri 0 ile 13 arasında olmalıdır: "
                f"{invalid_psm}"
            )

        if not 0 <= self.oem <= 3:
            raise ValueError("oem 0 ile 3 arasında olmalıdır.")

        if not 0.0 <= self.minimum_word_confidence <= 100.0:
            raise ValueError(
                "minimum_word_confidence 0 ile 100 arasında olmalıdır."
            )

        if not 0.0 <= self.minimum_result_confidence <= 1.0:
            raise ValueError(
                "minimum_result_confidence 0 ile 1 arasında olmalıdır."
            )

        if self.minimum_text_characters < 1:
            raise ValueError(
                "minimum_text_characters en az 1 olmalıdır."
            )

        if self.minimum_word_count < 1:
            raise ValueError("minimum_word_count en az 1 olmalıdır.")

        if self.upscale_factor < 1.0:
            raise ValueError("upscale_factor en az 1.0 olmalıdır.")

        if self.maximum_image_dimension < 512:
            raise ValueError(
                "maximum_image_dimension en az 512 olmalıdır."
            )

        if self.maximum_deskew_angle < 0:
            raise ValueError(
                "maximum_deskew_angle negatif olamaz."
            )

        allowed_variants = {
            "original",
            "grayscale",
            "otsu",
            "adaptive",
            "inverted_otsu",
        }
        unknown_variants = [
            item
            for item in self.preprocessing_variants
            if item not in allowed_variants
        ]
        if unknown_variants:
            raise ValueError(
                "Desteklenmeyen preprocessing variantları: "
                f"{unknown_variants}"
            )

        if not self.preprocessing_variants:
            raise ValueError(
                "En az bir preprocessing variantı tanımlanmalıdır."
            )

        if self.line_vertical_tolerance_ratio <= 0:
            raise ValueError(
                "line_vertical_tolerance_ratio pozitif olmalıdır."
            )

        if self.minimum_line_vertical_tolerance <= 0:
            raise ValueError(
                "minimum_line_vertical_tolerance pozitif olmalıdır."
            )

        if self.column_gap_height_ratio <= 0:
            raise ValueError(
                "column_gap_height_ratio pozitif olmalıdır."
            )

        if self.column_gap_character_ratio <= 0:
            raise ValueError(
                "column_gap_character_ratio pozitif olmalıdır."
            )

        if self.minimum_column_gap_pixels <= 0:
            raise ValueError(
                "minimum_column_gap_pixels pozitif olmalıdır."
            )

        if self.structured_minimum_rows < 1:
            raise ValueError(
                "structured_minimum_rows en az 1 olmalıdır."
            )

        if self.structured_minimum_columns < 1:
            raise ValueError(
                "structured_minimum_columns en az 1 olmalıdır."
            )

        if not 0.0 <= self.structured_row_ratio <= 1.0:
            raise ValueError(
                "structured_row_ratio 0 ile 1 arasında olmalıdır."
            )

        if not 0.0 <= self.early_stop_minimum_confidence <= 1.0:
            raise ValueError(
                "early_stop_minimum_confidence 0 ile 1 arasında olmalıdır."
            )

        if not 0.0 <= self.early_stop_minimum_score <= 1.0:
            raise ValueError(
                "early_stop_minimum_score 0 ile 1 arasında olmalıdır."
            )

        if self.early_stop_minimum_characters < 1:
            raise ValueError(
                "early_stop_minimum_characters en az 1 olmalıdır."
            )


@dataclass(slots=True, frozen=True)
class _OCRCandidate:
    """Tek preprocessing/PSM denemesinin ara sonucu."""

    preprocessing_variant: str
    psm_mode: int

    raw_text: str
    words: tuple[OCRWord, ...]
    lines: tuple[OCRLine, ...]
    reconstructed_rows: tuple[tuple[str, ...], ...]

    average_confidence: float
    score: float
    is_structured: bool

    metadata: Mapping[str, Any]


class WholeImageOCRExtractor:
    """
    Hücre yapısı bulunamayan tablo, metrik kartı ve infografik görsellerinin
    tamamına Tesseract OCR uygular.

    Akış:

    1. Görsel doğrulanır ve okunur.
    2. Gerekirse görsel büyütülür ve eğiklik düzeltilir.
    3. Birden fazla preprocessing varyantı oluşturulur.
    4. Her varyant birden fazla PSM modu ile OCR'dan geçirilir.
    5. Sözcük kutuları satırlara kümelenir.
    6. Büyük yatay boşluklardan olası sütunlar yeniden oluşturulur.
    7. Güven, metin kapsamı ve yapısallık skoruna göre en iyi sonuç seçilir.

    Bu sınıf hücre sınırlarına ihtiyaç duymaz. Tam bir tablo matrisi
    oluşturulamasa bile `raw_text` retrieval amacıyla korunur.
    """

    _WHITESPACE_RE = re.compile(r"\s+")
    _NUMERIC_TOKEN_RE = re.compile(
        r"""
        (?<!\w)
        [+-]?
        (?:
            \d{1,3}(?:[.\s,]\d{3})+
            |
            \d+
        )
        (?:[.,]\d+)?
        (?:\s*(?:%|‰|m3|m³|kg|ton|t|kwh|mwh|gwh|gj|tj|co2e|co₂e))?
        (?!\w)
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(
        self,
        *,
        config: WholeImageOCRConfig | None = None,
    ) -> None:
        self.config = config or WholeImageOCRConfig()

        if self.config.tesseract_command is not None:
            pytesseract.pytesseract.tesseract_cmd = str(
                Path(self.config.tesseract_command).expanduser()
            )

        self._validate_tesseract()

        self.language = self._resolve_language(
            self.config.language
        )

        if self.config.keep_debug_images:
            debug_directory = (
                self.config.debug_directory
                or Path("data/ocr_debug")
            )
            debug_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.debug_directory = debug_directory
        else:
            self.debug_directory = None

    def extract(
        self,
        visual: VisualAsset,
        *,
        update_visual: bool = True,
    ) -> WholeImageOCRResult:
        """
        `VisualAsset.image_path` üzerindeki görselin tamamına OCR uygular.

        `update_visual=True` olduğunda OCR sonucu `VisualAsset.mark_ocr_result`
        ile görsel nesnesine işlenir.
        """

        result = self.extract_from_path(
            image_path=visual.image_path,
            visual_id=visual.id,
            report_id=visual.report_id,
            page_number=visual.page_number,
        )

        if update_visual:
            visual.mark_ocr_result(
                result,
                minimum_confidence=(
                    self.config.minimum_result_confidence
                ),
            )

        return result

    def extract_many(
        self,
        visuals: Sequence[VisualAsset],
        *,
        update_visuals: bool = True,
        continue_on_error: bool = True,
    ) -> list[WholeImageOCRResult]:
        """Birden fazla görseli sırasıyla işler."""

        results: list[WholeImageOCRResult] = []

        for visual in visuals:
            try:
                result = self.extract(
                    visual,
                    update_visual=update_visuals,
                )
            except Exception as exc:
                visual.add_error(
                    f"Bütün-görüntü OCR başarısız: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.warning(
                    "Görsel %s bütün-görüntü OCR başarısız: %s",
                    visual.id,
                    exc,
                )

                if not continue_on_error:
                    raise

                continue

            results.append(result)

        return results

    def extract_from_path(
        self,
        *,
        image_path: str | Path,
        visual_id: UUID,
        report_id: UUID,
        page_number: int,
    ) -> WholeImageOCRResult:
        """Dosya yolundan OCR sonucu üretir."""

        resolved_path = Path(
            image_path
        ).expanduser().resolve()

        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"OCR görseli bulunamadı: {resolved_path}"
            )

        image = cv2.imread(
            str(resolved_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise WholeImageOCRError(
                f"Görsel OpenCV ile okunamadı: {resolved_path}"
            )

        prepared_image = self._prepare_base_image(image)
        image_hash = self._calculate_image_hash(image)

        candidates: list[_OCRCandidate] = []
        errors: list[str] = []

        variants = self._build_preprocessing_variants(
            prepared_image
        )

        early_stop_used = False
        attempted_combinations: list[str] = []

        for variant_name, variant_image in variants.items():
            self._save_debug_image(
                image=variant_image,
                image_hash=image_hash,
                suffix=variant_name,
            )

            for psm_mode in self.config.psm_modes:
                combination = f"{variant_name}/psm-{psm_mode}"
                attempted_combinations.append(combination)
                logger.info("Whole-image OCR deneniyor: %s", combination)

                try:
                    candidate = self._run_candidate(
                        image=variant_image,
                        preprocessing_variant=variant_name,
                        psm_mode=int(psm_mode),
                    )
                except Exception as exc:
                    errors.append(
                        f"{combination}: {type(exc).__name__}: {exc}"
                    )
                    logger.debug(
                        "OCR denemesi başarısız (%s): %s",
                        combination,
                        exc,
                    )
                    continue

                candidates.append(candidate)

                if self._should_stop_early(candidate):
                    early_stop_used = True
                    logger.info(
                        "Whole-image OCR erken durduruldu: %s, güven=%.3f, skor=%.3f",
                        combination,
                        candidate.average_confidence,
                        candidate.score,
                    )
                    break

            if early_stop_used:
                break

        if not candidates:
            error_detail = (
                "; ".join(errors[:5])
                if errors
                else "OCR adayı üretilemedi."
            )
            raise WholeImageOCRError(
                f"Hiçbir OCR denemesi tamamlanamadı: {error_detail}"
            )

        best = max(
            candidates,
            key=lambda item: item.score,
        )

        metadata = {
            "image_hash": image_hash,
            "candidate_count": len(candidates),
            "attempted_variant_count": len(variants),
            "attempted_psm_modes": list(self.config.psm_modes),
            "attempted_combinations": attempted_combinations,
            "early_stop_used": early_stop_used,
            "selected_score": round(best.score, 6),
            "selected_word_count": len(best.words),
            "selected_line_count": len(best.lines),
            "selected_numeric_token_count": (
                self._count_numeric_tokens(best.raw_text)
            ),
            "requested_language": self.config.language,
            "resolved_language": self.language,
            "errors": errors,
            **dict(best.metadata),
        }

        return WholeImageOCRResult(
            visual_id=visual_id,
            report_id=report_id,
            page_number=page_number,
            image_path=resolved_path,
            raw_text=best.raw_text,
            words=list(best.words),
            lines=list(best.lines),
            reconstructed_rows=[
                list(row)
                for row in best.reconstructed_rows
            ],
            average_confidence=round(
                best.average_confidence,
                6,
            ),
            preprocessing_variant=(
                best.preprocessing_variant
            ),
            psm_mode=best.psm_mode,
            language=self.language,
            is_structured=best.is_structured,
            metadata=metadata,
        )

    def _should_stop_early(self, candidate: _OCRCandidate) -> bool:
        if not self.config.enable_early_stop:
            return False

        character_count = len(re.sub(r"\s+", "", candidate.raw_text))

        if character_count < self.config.early_stop_minimum_characters:
            return False

        if candidate.average_confidence < self.config.early_stop_minimum_confidence:
            return False

        if candidate.score < self.config.early_stop_minimum_score:
            return False

        if (
            self.config.early_stop_require_structured
            and not candidate.is_structured
        ):
            return False

        return True

    def _run_candidate(
        self,
        *,
        image: np.ndarray,
        preprocessing_variant: str,
        psm_mode: int,
    ) -> _OCRCandidate:
        config = (
            f"--oem {self.config.oem} "
            f"--psm {psm_mode} "
            "-c preserve_interword_spaces=1"
        )

        data = pytesseract.image_to_data(
            image,
            lang=self.language,
            config=config,
            output_type=Output.DICT,
        )

        words = self._parse_words(data)

        lines = self._build_lines(words)

        raw_text = "\n".join(
            line.text
            for line in lines
        ).strip()

        reconstructed_rows = self._reconstruct_rows(
            lines
        )

        average_confidence = self._average_confidence(
            words
        )

        is_structured = self._is_structured(
            reconstructed_rows
        )

        score_metadata = self._score_candidate(
            raw_text=raw_text,
            words=words,
            lines=lines,
            reconstructed_rows=reconstructed_rows,
            average_confidence=average_confidence,
            is_structured=is_structured,
        )

        return _OCRCandidate(
            preprocessing_variant=preprocessing_variant,
            psm_mode=psm_mode,
            raw_text=raw_text,
            words=tuple(words),
            lines=tuple(lines),
            reconstructed_rows=tuple(
                tuple(row)
                for row in reconstructed_rows
            ),
            average_confidence=average_confidence,
            score=float(score_metadata["score"]),
            is_structured=is_structured,
            metadata=score_metadata,
        )

    def _parse_words(
        self,
        data: Mapping[str, Sequence[Any]],
    ) -> list[OCRWord]:
        texts = data.get("text", ())
        confidences = data.get("conf", ())
        left_values = data.get("left", ())
        top_values = data.get("top", ())
        width_values = data.get("width", ())
        height_values = data.get("height", ())
        block_values = data.get("block_num", ())
        line_values = data.get("line_num", ())

        item_count = min(
            len(texts),
            len(confidences),
            len(left_values),
            len(top_values),
            len(width_values),
            len(height_values),
        )

        words: list[OCRWord] = []

        for index in range(item_count):
            text = self._normalize_text(texts[index])

            if not text:
                continue

            confidence_raw = self._safe_float(
                confidences[index],
                default=-1.0,
            )

            if confidence_raw < self.config.minimum_word_confidence:
                continue

            left = max(
                int(self._safe_float(left_values[index])),
                0,
            )
            top = max(
                int(self._safe_float(top_values[index])),
                0,
            )
            width = max(
                int(self._safe_float(width_values[index])),
                0,
            )
            height = max(
                int(self._safe_float(height_values[index])),
                0,
            )

            if width <= 0 or height <= 0:
                continue

            block_index = (
                int(self._safe_float(block_values[index]))
                if index < len(block_values)
                else None
            )
            tesseract_line_index = (
                int(self._safe_float(line_values[index]))
                if index < len(line_values)
                else None
            )

            words.append(
                OCRWord(
                    text=text,
                    bounding_box=BoundingBox(
                        x1=float(left),
                        y1=float(top),
                        x2=float(left + width),
                        y2=float(top + height),
                    ),
                    confidence=max(
                        min(confidence_raw / 100.0, 1.0),
                        0.0,
                    ),
                    line_index=(
                        tesseract_line_index
                        if tesseract_line_index is not None
                        and tesseract_line_index >= 0
                        else None
                    ),
                    block_index=(
                        block_index
                        if block_index is not None
                        and block_index >= 0
                        else None
                    ),
                    metadata={
                        "tesseract_index": index,
                        "raw_confidence": confidence_raw,
                    },
                )
            )

        return words

    def _build_lines(
        self,
        words: Sequence[OCRWord],
    ) -> list[OCRLine]:
        if not words:
            return []

        heights = [
            word.bounding_box.height
            for word in words
        ]

        typical_height = (
            median(heights)
            if heights
            else self.config.minimum_line_vertical_tolerance
        )

        vertical_tolerance = max(
            typical_height
            * self.config.line_vertical_tolerance_ratio,
            self.config.minimum_line_vertical_tolerance,
        )

        sorted_words = sorted(
            words,
            key=lambda item: (
                self._center_y(item.bounding_box),
                item.bounding_box.x1,
            ),
        )

        line_groups: list[list[OCRWord]] = []
        line_centers: list[float] = []

        for word in sorted_words:
            word_center = self._center_y(
                word.bounding_box
            )

            best_index: int | None = None
            best_distance = math.inf

            for index, center in enumerate(line_centers):
                distance = abs(word_center - center)

                if (
                    distance <= vertical_tolerance
                    and distance < best_distance
                ):
                    best_index = index
                    best_distance = distance

            if best_index is None:
                line_groups.append([word])
                line_centers.append(word_center)
                continue

            line_groups[best_index].append(word)
            line_centers[best_index] = sum(
                self._center_y(item.bounding_box)
                for item in line_groups[best_index]
            ) / len(line_groups[best_index])

        ordered_groups = sorted(
            line_groups,
            key=lambda group: min(
                item.bounding_box.y1
                for item in group
            ),
        )

        lines: list[OCRLine] = []

        for line_index, group in enumerate(ordered_groups):
            ordered_words = sorted(
                group,
                key=lambda item: item.bounding_box.x1,
            )

            line_text = self._join_words(
                ordered_words
            )

            if not line_text:
                continue

            confidence = sum(
                item.confidence
                for item in ordered_words
            ) / len(ordered_words)

            lines.append(
                OCRLine(
                    line_index=line_index,
                    text=line_text,
                    bounding_box=self._merge_boxes(
                        item.bounding_box
                        for item in ordered_words
                    ),
                    words=list(ordered_words),
                    confidence=round(
                        confidence,
                        6,
                    ),
                    metadata={
                        "word_count": len(ordered_words),
                        "vertical_tolerance": round(
                            vertical_tolerance,
                            3,
                        ),
                    },
                )
            )

        return lines

    def _reconstruct_rows(
        self,
        lines: Sequence[OCRLine],
    ) -> list[list[str]]:
        rows: list[list[str]] = []

        for line in lines:
            words = sorted(
                line.words,
                key=lambda item: item.bounding_box.x1,
            )

            if not words:
                continue

            typical_height = median(
                word.bounding_box.height
                for word in words
            )

            character_widths = [
                word.bounding_box.width
                / max(len(word.text), 1)
                for word in words
            ]
            typical_character_width = (
                median(character_widths)
                if character_widths
                else typical_height / 2
            )

            split_threshold = max(
                typical_height
                * self.config.column_gap_height_ratio,
                typical_character_width
                * self.config.column_gap_character_ratio,
                self.config.minimum_column_gap_pixels,
            )

            cells: list[str] = []
            current_words: list[OCRWord] = [
                words[0]
            ]

            for previous, current in zip(
                words,
                words[1:],
            ):
                gap = (
                    current.bounding_box.x1
                    - previous.bounding_box.x2
                )

                if gap >= split_threshold:
                    cells.append(
                        self._join_words(current_words)
                    )
                    current_words = [current]
                else:
                    current_words.append(current)

            if current_words:
                cells.append(
                    self._join_words(current_words)
                )

            normalized_cells = [
                self._normalize_text(value)
                for value in cells
                if self._normalize_text(value)
            ]

            if normalized_cells:
                rows.append(normalized_cells)

        return rows

    def _is_structured(
        self,
        rows: Sequence[Sequence[str]],
    ) -> bool:
        if len(rows) < self.config.structured_minimum_rows:
            return False

        column_counts = [
            len(row)
            for row in rows
            if row
        ]

        if not column_counts:
            return False

        qualifying_rows = sum(
            count >= self.config.structured_minimum_columns
            for count in column_counts
        )

        qualifying_ratio = (
            qualifying_rows / len(column_counts)
        )

        most_common_column_count, frequency = Counter(
            column_counts
        ).most_common(1)[0]

        consistency_ratio = (
            frequency / len(column_counts)
        )

        return (
            most_common_column_count
            >= self.config.structured_minimum_columns
            and qualifying_ratio
            >= self.config.structured_row_ratio
            and consistency_ratio >= 0.35
        )

    def _score_candidate(
        self,
        *,
        raw_text: str,
        words: Sequence[OCRWord],
        lines: Sequence[OCRLine],
        reconstructed_rows: Sequence[Sequence[str]],
        average_confidence: float,
        is_structured: bool,
    ) -> dict[str, Any]:
        character_count = len(
            re.sub(r"\s+", "", raw_text)
        )

        word_count = len(words)
        line_count = len(lines)
        numeric_token_count = self._count_numeric_tokens(
            raw_text
        )

        valid_text = (
            character_count
            >= self.config.minimum_text_characters
            and word_count >= self.config.minimum_word_count
        )

        text_coverage_score = min(
            character_count / 250.0,
            1.0,
        )

        word_coverage_score = min(
            word_count / 40.0,
            1.0,
        )

        line_coverage_score = min(
            line_count / 12.0,
            1.0,
        )

        numeric_score = min(
            numeric_token_count / 8.0,
            1.0,
        )

        structured_score = (
            1.0
            if is_structured
            else min(
                sum(
                    len(row) >= 2
                    for row in reconstructed_rows
                )
                / max(len(reconstructed_rows), 1),
                1.0,
            )
        )

        gibberish_ratio = self._gibberish_ratio(
            raw_text
        )

        score = (
            average_confidence * 0.42
            + text_coverage_score * 0.18
            + word_coverage_score * 0.12
            + line_coverage_score * 0.08
            + numeric_score * 0.12
            + structured_score * 0.08
            - gibberish_ratio * 0.20
        )

        if not valid_text:
            score -= 0.35

        if average_confidence < 0.20:
            score -= 0.15

        return {
            "score": round(score, 6),
            "character_count": character_count,
            "word_count": word_count,
            "line_count": line_count,
            "numeric_token_count": numeric_token_count,
            "text_coverage_score": round(
                text_coverage_score,
                6,
            ),
            "word_coverage_score": round(
                word_coverage_score,
                6,
            ),
            "line_coverage_score": round(
                line_coverage_score,
                6,
            ),
            "numeric_score": round(
                numeric_score,
                6,
            ),
            "structured_score": round(
                structured_score,
                6,
            ),
            "gibberish_ratio": round(
                gibberish_ratio,
                6,
            ),
            "valid_text": valid_text,
        }

    def _prepare_base_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        if image.ndim != 3:
            raise WholeImageOCRError(
                "Görsel beklenen BGR renk formatında değil."
            )

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            raise WholeImageOCRError(
                "Görsel boyutları geçersiz."
            )

        scale = self.config.upscale_factor

        longest_dimension = max(
            width * scale,
            height * scale,
        )

        if longest_dimension > self.config.maximum_image_dimension:
            scale = (
                self.config.maximum_image_dimension
                / max(width, height)
            )

        if scale > 1.0:
            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        if self.config.apply_deskew:
            image = self._deskew_image(image)

        return image

    def _build_preprocessing_variants(
        self,
        image: np.ndarray,
    ) -> dict[str, np.ndarray]:
        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        if self.config.apply_denoising:
            grayscale = cv2.fastNlMeansDenoising(
                grayscale,
                None,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        if self.config.apply_sharpening:
            blurred = cv2.GaussianBlur(
                grayscale,
                (0, 0),
                sigmaX=1.0,
            )
            grayscale = cv2.addWeighted(
                grayscale,
                1.6,
                blurred,
                -0.6,
                0,
            )

        variants: dict[str, np.ndarray] = {}

        for variant in self.config.preprocessing_variants:
            if variant == "original":
                variants[variant] = image.copy()

            elif variant == "grayscale":
                variants[variant] = grayscale.copy()

            elif variant == "otsu":
                _, thresholded = cv2.threshold(
                    grayscale,
                    0,
                    255,
                    cv2.THRESH_BINARY
                    | cv2.THRESH_OTSU,
                )
                variants[variant] = thresholded

            elif variant == "inverted_otsu":
                _, thresholded = cv2.threshold(
                    grayscale,
                    0,
                    255,
                    cv2.THRESH_BINARY_INV
                    | cv2.THRESH_OTSU,
                )
                variants[variant] = thresholded

            elif variant == "adaptive":
                block_size = self._adaptive_block_size(
                    grayscale
                )
                adaptive = cv2.adaptiveThreshold(
                    grayscale,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block_size,
                    15,
                )
                variants[variant] = adaptive

        return variants

    def _deskew_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        _, binary = cv2.threshold(
            grayscale,
            0,
            255,
            cv2.THRESH_BINARY_INV
            | cv2.THRESH_OTSU,
        )

        coordinates = np.column_stack(
            np.where(binary > 0)
        )

        if coordinates.shape[0] < 20:
            return image

        angle = cv2.minAreaRect(
            coordinates[:, ::-1].astype(np.float32)
        )[-1]

        if angle < -45:
            angle = 90 + angle

        if abs(angle) > self.config.maximum_deskew_angle:
            return image

        if abs(angle) < 0.15:
            return image

        height, width = image.shape[:2]
        center = (
            width / 2.0,
            height / 2.0,
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )

        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _join_words(
        self,
        words: Sequence[OCRWord],
    ) -> str:
        if not words:
            return ""

        parts: list[str] = []

        for word in words:
            normalized = self._normalize_text(
                word.text
            )

            if not normalized:
                continue

            parts.append(normalized)

        return self._normalize_text(
            " ".join(parts)
        )

    def _average_confidence(
        self,
        words: Sequence[OCRWord],
    ) -> float:
        if not words:
            return 0.0

        total_weight = 0.0
        weighted_confidence = 0.0

        for word in words:
            weight = max(
                len(word.text),
                1,
            )
            total_weight += weight
            weighted_confidence += (
                word.confidence * weight
            )

        return weighted_confidence / max(
            total_weight,
            1.0,
        )

    def _gibberish_ratio(
        self,
        text: str,
    ) -> float:
        compact = re.sub(
            r"\s+",
            "",
            text,
        )

        if not compact:
            return 1.0

        meaningful = sum(
            character.isalnum()
            or character in ".,:%‰₺$€£+-/()[]{}²³₂"
            for character in compact
        )

        return 1.0 - (
            meaningful / len(compact)
        )

    def _count_numeric_tokens(
        self,
        text: str,
    ) -> int:
        return len(
            self._NUMERIC_TOKEN_RE.findall(text)
        )

    def _resolve_language(
        self,
        requested_language: str,
    ) -> str:
        try:
            available_languages = set(
                pytesseract.get_languages(
                    config="",
                )
            )
        except Exception:
            return requested_language

        requested = [
            item.strip()
            for item in requested_language.split("+")
            if item.strip()
        ]

        resolved = [
            item
            for item in requested
            if item in available_languages
        ]

        if resolved:
            return "+".join(resolved)

        if "eng" in available_languages:
            logger.warning(
                "İstenen Tesseract dilleri bulunamadı (%s). "
                "'eng' kullanılacak.",
                requested_language,
            )
            return "eng"

        if available_languages:
            fallback = sorted(
                available_languages
            )[0]
            logger.warning(
                "İstenen Tesseract dilleri bulunamadı (%s). "
                "%s kullanılacak.",
                requested_language,
                fallback,
            )
            return fallback

        return requested_language

    def _validate_tesseract(self) -> None:
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise WholeImageOCRError(
                "Tesseract çalıştırılamadı. Tesseract uygulamasının "
                "kurulu ve PATH üzerinde olduğundan emin olun."
            ) from exc

    def _save_debug_image(
        self,
        *,
        image: np.ndarray,
        image_hash: str,
        suffix: str,
    ) -> None:
        if self.debug_directory is None:
            return

        output_path = (
            self.debug_directory
            / f"{image_hash[:16]}_{suffix}.png"
        )

        success = cv2.imwrite(
            str(output_path),
            image,
        )

        if not success:
            logger.warning(
                "OCR debug görseli kaydedilemedi: %s",
                output_path,
            )

    @staticmethod
    def _adaptive_block_size(
        image: np.ndarray,
    ) -> int:
        shortest_dimension = min(
            image.shape[:2]
        )

        proposed = max(
            int(shortest_dimension / 25),
            15,
        )

        if proposed % 2 == 0:
            proposed += 1

        return min(
            proposed,
            101,
        )

    @staticmethod
    def _calculate_image_hash(
        image: np.ndarray,
    ) -> str:
        return hashlib.sha256(
            image.tobytes()
        ).hexdigest()

    @classmethod
    def _normalize_text(
        cls,
        value: Any,
    ) -> str:
        if value is None:
            return ""

        return cls._WHITESPACE_RE.sub(
            " ",
            str(value).strip(),
        )

    @staticmethod
    def _safe_float(
        value: Any,
        *,
        default: float = 0.0,
    ) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default

        if math.isnan(parsed) or math.isinf(parsed):
            return default

        return parsed

    @staticmethod
    def _center_y(
        box: BoundingBox,
    ) -> float:
        return (
            box.y1 + box.y2
        ) / 2.0

    @staticmethod
    def _merge_boxes(
        boxes: Iterable[BoundingBox],
    ) -> BoundingBox:
        box_list = list(boxes)

        if not box_list:
            raise ValueError(
                "Birleştirilecek bounding box bulunamadı."
            )

        return BoundingBox(
            x1=min(box.x1 for box in box_list),
            y1=min(box.y1 for box in box_list),
            x2=max(box.x2 for box in box_list),
            y2=max(box.y2 for box in box_list),
        )