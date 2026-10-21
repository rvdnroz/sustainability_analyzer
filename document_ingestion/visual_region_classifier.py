from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(slots=True, frozen=True)
class VisualRegionCandidate:
    page_number: int
    source_element_id: str
    bounding_box: Any
    region_type: str
    confidence: float
    caption: str | None
    nearby_text: str | None
    metadata: dict[str, Any]


class VisualRegionClassifier:
    """Layout IMAGE/FIGURE öğelerini tablo hattından bağımsız sınıflandırır."""

    def __init__(
        self,
        *,
        minimum_area_ratio: float = 0.025,
        minimum_width_ratio: float = 0.16,
        minimum_height_ratio: float = 0.08,
        nearby_vertical_distance: float = 80.0,
    ) -> None:
        self.minimum_area_ratio = minimum_area_ratio
        self.minimum_width_ratio = minimum_width_ratio
        self.minimum_height_ratio = minimum_height_ratio
        self.nearby_vertical_distance = nearby_vertical_distance

    def classify_layouts(self, layouts: Sequence[Any]) -> list[VisualRegionCandidate]:
        result: list[VisualRegionCandidate] = []
        for layout in layouts:
            result.extend(self.classify_layout(layout))
        return result

    def classify_layout(self, layout: Any) -> list[VisualRegionCandidate]:
        page_number = int(getattr(layout, "page_number", 0) or 0)
        page_width = float(getattr(layout, "width_points", 0.0) or 0.0)
        page_height = float(getattr(layout, "height_points", 0.0) or 0.0)
        if page_number <= 0 or page_width <= 0 or page_height <= 0:
            return []

        elements = list(getattr(layout, "elements", ()) or ())
        visuals = [
            item for item in elements
            if self._enum_value(getattr(item, "element_type", "")).casefold()
            in {"image", "figure"}
        ]
        texts = [
            item for item in elements
            if str(getattr(item, "text", "") or "").strip()
        ]

        page_text = self._normalize(
            "\n".join(
                str(getattr(item, "text", "") or "").strip()
                for item in texts
                if str(getattr(item, "text", "") or "").strip()
            )
        )
        page_is_reporting_framework_overview = (
            self._is_reporting_framework_overview(page_text)
        )

        candidates: list[VisualRegionCandidate] = []
        for index, visual in enumerate(visuals):
            box = getattr(visual, "bounding_box", None)
            if box is None:
                continue

            width_ratio = float(getattr(box, "width", 0.0) or 0.0) / max(page_width, 1.0)
            height_ratio = float(getattr(box, "height", 0.0) or 0.0) / max(page_height, 1.0)
            area_ratio = float(getattr(box, "area", 0.0) or 0.0) / max(page_width * page_height, 1.0)

            if (
                width_ratio < self.minimum_width_ratio
                or height_ratio < self.minimum_height_ratio
                or area_ratio < self.minimum_area_ratio
                or area_ratio > 0.92
            ):
                continue

            nearby = self._nearby_text(box, texts)
            normalized = self._normalize("\n".join(nearby))
            scores = self._score(
                normalized,
                source_type=self._enum_value(getattr(visual, "element_type", "")).casefold(),
                width_ratio=width_ratio,
                height_ratio=height_ratio,
            )
            region_type, confidence = self._choose(scores)

            # Raporlama standardı/logo genel bakış sayfaları grafik değildir.
            # Yakındaki tek bir tarih veya sayı, bu sayfalardaki logo kümelerini
            # yanlışlıkla chart sınıfına taşımamalıdır.
            if page_is_reporting_framework_overview:
                region_type = "infographic"
                confidence = max(
                    float(scores.get("infographic", 0.0)),
                    0.92,
                )
                scores["chart"] = 0.0
                scores["infographic"] = confidence

            if region_type == "decorative":
                continue

            candidates.append(
                VisualRegionCandidate(
                    page_number=page_number,
                    source_element_id=str(
                        getattr(visual, "element_id", "")
                        or f"page-{page_number}-visual-{index + 1}"
                    ),
                    bounding_box=box,
                    region_type=region_type,
                    confidence=confidence,
                    caption=nearby[0] if nearby else None,
                    nearby_text="\n".join(nearby[:6]).strip() or None,
                    metadata={
                        "visual_region_scores": scores,
                        "source_layout_type": self._enum_value(
                            getattr(visual, "element_type", "")
                        ),
                        "width_ratio": round(width_ratio, 5),
                        "height_ratio": round(height_ratio, 5),
                        "area_ratio": round(area_ratio, 5),
                        "chart_gate_version": "strict_v3",
                        "page_reporting_framework_overview": (
                            page_is_reporting_framework_overview
                        ),
                    },
                )
            )

        return self._deduplicate(candidates)

    def _score(
        self,
        text: str,
        *,
        source_type: str,
        width_ratio: float,
        height_ratio: float,
    ) -> dict[str, float]:
        """
        Layout çevresindeki metinden görsel türü skoru üretir.

        Grafik kararı yalnızca genel ESG kelimelerine dayanmaz. En az bir
        yapısal grafik sinyali aranır: açık grafik terimi, çoklu yıl +
        çoklu sayı, eksen/seri dili veya chart-specific sözcük kombinasyonu.
        """

        scores = {
            "chart": 0.0,
            "metric_card": 0.0,
            "infographic": 0.0,
            "figure": 0.0,
            "decorative": 0.0,
            "unknown": 0.0,
        }

        explicit_chart = (
            "chart", "graph", "bar chart", "line chart", "pie chart",
            "stacked bar", "column chart", "donut chart", "area chart",
            "grafik", "cizgi grafik", "sutun grafik", "pasta grafik",
        )
        axis_terms = (
            "x axis", "y axis", "axis", "legend", "series",
            "data labels", "trend line", "bars", "columns",
            "eksen", "lejand", "seri",
        )
        chart_context = (
            "trend", "distribution", "breakdown", "by year", "yearly",
            "annual trend", "target vs actual", "baseline",
            "female male total", "scope 1 scope 2", "scope 1 scope 2 scope 3",
            "dagilim", "egilim", "yillara gore",
        )
        infographic_terms = (
            "infographic", "roadmap", "timeline", "value chain",
            "process flow", "journey", "infografik", "framework",
            "reporting standards", "standards and frameworks",
        )
        metric_terms = (
            "kpi", "at a glance", "in numbers", "highlights",
            "key performance indicator", "rakamlarla", "one bakis",
        )
        logo_framework_terms = (
            "ungc", "sdgs", "weps", "iirc", "sasb", "gri", "tcfd",
            "reporting standards", "frameworks", "standards and frameworks",
            "uyumlu oldugumuz raporlama standartlari",
        )

        explicit_chart_hit = any(term in text for term in explicit_chart)
        axis_hit_count = sum(term in text for term in axis_terms)
        context_hit_count = sum(term in text for term in chart_context)
        framework_hit_count = sum(term in text for term in logo_framework_terms)

        years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
        numbers = re.findall(
            r"(?<!\w)[+-]?(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)"
            r"(?:[.,]\d+)?(?:\s*%)?(?!\w)",
            text,
        )

        if source_type == "figure":
            scores["figure"] += 0.32

        if explicit_chart_hit:
            scores["chart"] += 0.62

        if axis_hit_count:
            scores["chart"] += min(axis_hit_count * 0.18, 0.36)

        if context_hit_count:
            scores["chart"] += min(context_hit_count * 0.10, 0.30)

        # Çoklu yıl tek başına yeterli değildir; sayısal yoğunlukla birlikte
        # gerçek grafik sinyali sayılır.
        if len(years) >= 3 and len(numbers) >= 6:
            scores["chart"] += 0.30
        elif len(years) >= 2 and len(numbers) >= 5:
            scores["chart"] += 0.18

        if len(numbers) >= 10:
            scores["chart"] += 0.10
            scores["metric_card"] += 0.08
        elif len(numbers) >= 5:
            scores["metric_card"] += 0.05

        if width_ratio >= 0.55 and 0.14 <= height_ratio <= 0.68:
            scores["chart"] += 0.08

        if any(term in text for term in infographic_terms):
            scores["infographic"] += 0.72

        if any(term in text for term in metric_terms):
            scores["metric_card"] += 0.66

        # Logo/standart çerçevesi sayfalarını grafik olmaktan çıkar.
        if framework_hit_count:
            scores["infographic"] += min(0.55 + framework_hit_count * 0.08, 0.90)
            scores["chart"] -= min(0.45 + framework_hit_count * 0.05, 0.75)
            scores["figure"] += 0.18

        if not text:
            scores["figure"] += 0.36
            scores["decorative"] += 0.22

        # Grafik için yapısal kanıt yoksa chart skorunu baskıla.
        has_structural_chart_signal = (
            explicit_chart_hit
            or axis_hit_count >= 1
            or (len(years) >= 3 and len(numbers) >= 6)
            or context_hit_count >= 2
        )
        if not has_structural_chart_signal:
            scores["chart"] = min(scores["chart"], 0.44)

        if max(scores.values()) < 0.35:
            scores["unknown"] = 0.40

        return {
            key: round(min(max(value, 0.0), 1.0), 4)
            for key, value in scores.items()
        }

    @staticmethod
    def _choose(scores: dict[str, float]) -> tuple[str, float]:
        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        best_type, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        thresholds = {
            "chart": 0.74,
            "metric_card": 0.66,
            "infographic": 0.68,
            "figure": 0.50,
            "decorative": 0.60,
            "unknown": 0.0,
        }

        if best_score < thresholds.get(best_type, 0.54):
            return "unknown", round(best_score, 4)

        # Chart için daha geniş ayrım marjı istenir; logo/infographic
        # sayfalarının yanlışlıkla chart seçilmesini azaltır.
        required_margin = 0.14 if best_type == "chart" else 0.08
        if best_score - second_score < required_margin:
            return "unknown", round(best_score, 4)

        return best_type, round(best_score, 4)

    def _nearby_text(self, visual_box: Any, text_elements: Sequence[Any]) -> list[str]:
        scored: list[tuple[float, str]] = []
        for element in text_elements:
            text = str(getattr(element, "text", "") or "").strip()
            box = getattr(element, "bounding_box", None)
            if not text or box is None:
                continue
            vertical_distance = min(
                abs(float(getattr(box, "y1", 0.0)) - float(getattr(visual_box, "y0", 0.0))),
                abs(float(getattr(visual_box, "y1", 0.0)) - float(getattr(box, "y0", 0.0))),
            )
            horizontal_overlap = self._horizontal_overlap_ratio(visual_box, box)
            if vertical_distance <= self.nearby_vertical_distance and horizontal_overlap >= 0.20:
                scored.append((vertical_distance - horizontal_overlap * 20.0, text))
        scored.sort(key=lambda item: item[0])
        return [text for _, text in scored[:8]]

    @staticmethod
    def _horizontal_overlap_ratio(first: Any, second: Any) -> float:
        x0 = max(float(getattr(first, "x0", 0.0)), float(getattr(second, "x0", 0.0)))
        x1 = min(float(getattr(first, "x1", 0.0)), float(getattr(second, "x1", 0.0)))
        if x1 <= x0:
            return 0.0
        width_first = max(float(getattr(first, "width", 0.0)), 1.0)
        width_second = max(float(getattr(second, "width", 0.0)), 1.0)
        return min((x1 - x0) / min(width_first, width_second), 1.0)

    @staticmethod
    def _is_reporting_framework_overview(text: str) -> bool:
        """
        Raporlama standartları, çerçeveler ve logo kümelerinden oluşan
        genel bakış sayfalarını belirler.
        """

        strong_phrases = (
            "reporting standards and frameworks",
            "reporting standards",
            "standards and frameworks",
            "frameworks we comply with",
            "uyumlu oldugumuz raporlama standartlari",
            "raporlama standartlari ve cerceveleri",
        )
        framework_tokens = (
            "ungc",
            "sdgs",
            "weps",
            "iirc",
            "sasb",
            "gri",
            "tcfd",
            "tsrs",
        )

        if any(phrase in text for phrase in strong_phrases):
            return True

        token_hits = sum(
            1
            for token in framework_tokens
            if re.search(rf"\b{re.escape(token)}\b", text)
        )
        return token_hits >= 3

    @staticmethod
    def _normalize(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(value or "").strip().casefold())
        ascii_text = "".join(
            character for character in normalized
            if not unicodedata.combining(character)
        ).replace("ı", "i")
        return re.sub(r"[^a-z0-9%²³]+", " ", ascii_text).strip()

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value) or "")

    @classmethod
    def _deduplicate(cls, candidates: Sequence[VisualRegionCandidate]) -> list[VisualRegionCandidate]:
        ordered = sorted(candidates, key=lambda item: item.confidence, reverse=True)
        result: list[VisualRegionCandidate] = []
        for candidate in ordered:
            if any(
                candidate.page_number == existing.page_number
                and cls._iou(candidate.bounding_box, existing.bounding_box) >= 0.85
                for existing in result
            ):
                continue
            result.append(candidate)
        return sorted(
            result,
            key=lambda item: (
                item.page_number,
                float(getattr(item.bounding_box, "y0", 0.0)),
                float(getattr(item.bounding_box, "x0", 0.0)),
            ),
        )

    @staticmethod
    def _iou(first: Any, second: Any) -> float:
        x0 = max(float(getattr(first, "x0", 0.0)), float(getattr(second, "x0", 0.0)))
        y0 = max(float(getattr(first, "y0", 0.0)), float(getattr(second, "y0", 0.0)))
        x1 = min(float(getattr(first, "x1", 0.0)), float(getattr(second, "x1", 0.0)))
        y1 = min(float(getattr(first, "y1", 0.0)), float(getattr(second, "y1", 0.0)))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        intersection = (x1 - x0) * (y1 - y0)
        first_area = max(float(getattr(first, "area", 0.0)), 0.0)
        second_area = max(float(getattr(second, "area", 0.0)), 0.0)
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0


__all__ = ["VisualRegionCandidate", "VisualRegionClassifier"]