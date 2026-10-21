from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

try:
    import pymupdf
except ImportError:  # Eski PyMuPDF sürümleri için
    import fitz as pymupdf  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

ImageFormat = Literal["png", "jpeg"]


class PageRenderError(RuntimeError):
    """PDF sayfaları görüntüye dönüştürülemediğinde yükseltilir."""


@dataclass(slots=True, frozen=True)
class RenderedPage:
    """Bir PDF sayfasına ait render çıktısı ve temel metadata."""

    page_index: int
    page_number: int
    image_path: Path
    image_format: ImageFormat
    width_points: float
    height_points: float
    width_pixels: int
    height_pixels: int
    dpi: int
    rotation: int
    file_size_bytes: int
    render_time_ms: float
    reused_existing_file: bool = False

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        return data


class PageRenderer:
    """PDF sayfalarını PNG veya JPEG görüntülerine dönüştürür."""

    def __init__(
        self,
        *,
        output_directory: str | Path = "data/rendered_pages",
        dpi: int = 150,
        image_format: ImageFormat = "png",
        jpeg_quality: int = 90,
        include_annotations: bool = False,
        max_pixels_per_page: int = 50_000_000,
    ) -> None:
        if not 72 <= dpi <= 600:
            raise ValueError("dpi 72 ile 600 arasında olmalıdır.")
        if image_format not in {"png", "jpeg"}:
            raise ValueError("image_format 'png' veya 'jpeg' olmalıdır.")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality 1 ile 100 arasında olmalıdır.")
        if max_pixels_per_page <= 0:
            raise ValueError("max_pixels_per_page pozitif olmalıdır.")

        self.output_directory = (
            Path(output_directory)
            .expanduser()
            .resolve()
        )
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.dpi = dpi
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.include_annotations = include_annotations
        self.max_pixels_per_page = max_pixels_per_page

    def render(
        self,
        pdf_path: str | Path,
        output_dir: str | Path | None = None,
        *,
        page_numbers: Iterable[int] | None = None,
        overwrite: bool = False,
        write_manifest: bool = True,
    ) -> list[RenderedPage]:
        """
        PDF sayfalarını görüntüye dönüştürür.

        ``page_numbers`` 1 tabanlıdır. None verilirse bütün sayfalar
        işlenir. ``output_dir`` verilmezse çıktı klasörü otomatik olarak
        ``output_directory / PDF_ADI`` biçiminde oluşturulur.
        """
        pdf_path = Path(pdf_path).expanduser().resolve()
        self._validate_pdf_path(pdf_path)

        if output_dir is None:
            resolved_output_dir = self.output_directory / pdf_path.stem
        else:
            resolved_output_dir = Path(output_dir).expanduser().resolve()

        resolved_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            document = pymupdf.open(pdf_path)
        except Exception as exc:
            raise PageRenderError(
                f"PDF açılamadı: {pdf_path}",
            ) from exc

        try:
            if document.needs_pass:
                raise PageRenderError(
                    "Parola korumalı PDF render edilemez.",
                )
            if document.page_count == 0:
                raise PageRenderError(
                    "PDF içinde sayfa bulunamadı.",
                )

            selected_pages = self._normalize_pages(
                page_numbers,
                document.page_count,
            )

            results = [
                self._render_page(
                    page=document.load_page(page_number - 1),
                    output_dir=resolved_output_dir,
                    overwrite=overwrite,
                )
                for page_number in selected_pages
            ]

            if write_manifest:
                self._write_manifest(
                    pdf_path=pdf_path,
                    output_dir=resolved_output_dir,
                    total_page_count=document.page_count,
                    results=results,
                )

            logger.info(
                "%d sayfa render edildi: %s",
                len(results),
                resolved_output_dir,
            )
            return results
        finally:
            document.close()

    def _render_page(
        self,
        *,
        page: pymupdf.Page,
        output_dir: Path,
        overwrite: bool,
    ) -> RenderedPage:
        """Tek bir PDF sayfasını görüntü dosyasına dönüştürür."""
        page_index = int(page.number)
        page_number = page_index + 1
        extension = "jpg" if self.image_format == "jpeg" else "png"
        image_path = output_dir / f"page_{page_number:04d}.{extension}"

        if image_path.exists() and not overwrite:
            try:
                pixmap = pymupdf.Pixmap(str(image_path))
            except Exception as exc:
                raise PageRenderError(
                    f"Mevcut görüntü okunamadı: {image_path}",
                ) from exc

            return self._build_result(
                page=page,
                image_path=image_path,
                pixmap=pixmap,
                render_time_ms=0.0,
                reused=True,
            )

        estimated_pixels = int(
            page.rect.width
            * self.dpi
            / 72
            * page.rect.height
            * self.dpi
            / 72
        )
        if estimated_pixels > self.max_pixels_per_page:
            raise PageRenderError(
                f"Sayfa {page_number} çok büyük; DPI değerini düşürün.",
            )

        started_at = time.perf_counter()
        temporary_path = image_path.with_suffix(
            image_path.suffix + ".part",
        )

        try:
            pixmap = page.get_pixmap(
                dpi=self.dpi,
                colorspace=pymupdf.csRGB,
                alpha=False,
                annots=self.include_annotations,
            )

            if pixmap.width * pixmap.height > self.max_pixels_per_page:
                raise PageRenderError(
                    f"Sayfa {page_number} render sonucu "
                    "piksel sınırını aştı.",
                )

            image_bytes = (
                pixmap.tobytes(
                    "jpeg",
                    jpg_quality=self.jpeg_quality,
                )
                if self.image_format == "jpeg"
                else pixmap.tobytes("png")
            )

            temporary_path.write_bytes(image_bytes)
            os.replace(temporary_path, image_path)
        except PageRenderError:
            temporary_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            raise PageRenderError(
                f"Sayfa {page_number} render edilemedi.",
            ) from exc

        return self._build_result(
            page=page,
            image_path=image_path,
            pixmap=pixmap,
            render_time_ms=round(
                (time.perf_counter() - started_at) * 1000,
                3,
            ),
            reused=False,
        )

    def _build_result(
        self,
        *,
        page: pymupdf.Page,
        image_path: Path,
        pixmap: pymupdf.Pixmap,
        render_time_ms: float,
        reused: bool,
    ) -> RenderedPage:
        page_index = int(page.number)

        return RenderedPage(
            page_index=page_index,
            page_number=page_index + 1,
            image_path=image_path,
            image_format=self.image_format,
            width_points=float(page.rect.width),
            height_points=float(page.rect.height),
            width_pixels=int(pixmap.width),
            height_pixels=int(pixmap.height),
            dpi=self.dpi,
            rotation=int(page.rotation),
            file_size_bytes=image_path.stat().st_size,
            render_time_ms=render_time_ms,
            reused_existing_file=reused,
        )

    def _write_manifest(
        self,
        *,
        pdf_path: Path,
        output_dir: Path,
        total_page_count: int,
        results: list[RenderedPage],
    ) -> None:
        manifest = {
            "source_pdf": str(pdf_path),
            "total_page_count": total_page_count,
            "rendered_page_count": len(results),
            "dpi": self.dpi,
            "image_format": self.image_format,
            "pages": [result.to_dict() for result in results],
        }

        manifest_path = output_dir / "render_manifest.json"
        temporary_path = manifest_path.with_suffix(".json.part")

        try:
            temporary_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(temporary_path, manifest_path)
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            raise PageRenderError(
                "Render manifesti yazılamadı.",
            ) from exc

    @staticmethod
    def _normalize_pages(
        page_numbers: Iterable[int] | None,
        page_count: int,
    ) -> list[int]:
        if page_numbers is None:
            return list(range(1, page_count + 1))

        pages = list(
            dict.fromkeys(
                int(number)
                for number in page_numbers
            ),
        )
        invalid = [
            number
            for number in pages
            if not 1 <= number <= page_count
        ]

        if not pages:
            raise ValueError(
                "En az bir sayfa numarası verilmelidir.",
            )
        if invalid:
            raise ValueError(
                f"Geçersiz sayfalar: {invalid}. "
                f"Geçerli aralık 1-{page_count}.",
            )

        return pages

    @staticmethod
    def _validate_pdf_path(
        pdf_path: Path,
    ) -> None:
        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF bulunamadı: {pdf_path}",
            )
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Yalnızca .pdf uzantılı dosyalar desteklenir.",
            )


def render_pdf_pages(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 150,
    image_format: ImageFormat = "png",
    page_numbers: Iterable[int] | None = None,
    overwrite: bool = False,
) -> list[RenderedPage]:
    """PageRenderer için kısa kullanım arayüzü."""
    return PageRenderer(
        output_directory=output_dir,
        dpi=dpi,
        image_format=image_format,
    ).render(
        pdf_path=pdf_path,
        output_dir=output_dir,
        page_numbers=page_numbers,
        overwrite=overwrite,
    )


__all__ = [
    "PageRenderError",
    "PageRenderer",
    "RenderedPage",
    "render_pdf_pages",
]
