from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence


logger = logging.getLogger(__name__)


class FileStoreError(RuntimeError):
    """Dosya deposu işlemleri başarısız olduğunda yükseltilir."""


class ArtifactType(StrEnum):
    """Sistemde saklanabilen temel artifact türleri."""

    PDF = "pdf"
    RENDER = "render"
    OCR = "ocr"
    LAYOUT = "layout"
    CLASSIFICATION = "classification"
    TABLE = "table"
    CHUNK = "chunk"
    EMBEDDING = "embedding"
    EVIDENCE = "evidence"
    ANSWER = "answer"
    EXPORT = "export"
    CACHE = "cache"
    LOG = "log"
    TEMP = "temp"
    OTHER = "other"


@dataclass(slots=True, frozen=True)
class StoredArtifact:
    """Dosya deposuna yazılan tek bir artifact kaydı."""

    artifact_type: ArtifactType
    path: Path
    relative_path: Path

    file_name: str
    file_extension: str
    file_size_bytes: int
    sha256: str

    created_at: datetime

    company_slug: str | None = None
    report_year: int | None = None
    report_id: str | None = None
    page_number: int | None = None

    metadata_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type.value,
            "path": str(self.path),
            "relative_path": str(self.relative_path),
            "file_name": self.file_name,
            "file_extension": self.file_extension,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "created_at": self.created_at.isoformat(),
            "company_slug": self.company_slug,
            "report_year": self.report_year,
            "report_id": self.report_id,
            "page_number": self.page_number,
            "metadata_path": (
                str(self.metadata_path)
                if self.metadata_path is not None
                else None
            ),
        }


@dataclass(slots=True, frozen=True)
class ReportPaths:
    """Tek bir şirket-yıl raporu için kullanılan klasör yolları."""

    report_root: Path
    pdf_dir: Path
    render_dir: Path
    ocr_dir: Path
    layout_dir: Path
    classification_dir: Path
    table_dir: Path
    chunk_dir: Path
    evidence_dir: Path
    answer_dir: Path
    temp_dir: Path

    def create(self) -> None:
        for path in asdict(self).values():
            Path(path).mkdir(
                parents=True,
                exist_ok=True,
            )


class FileStore:
    """
    Sustainability Analyzer dosya ve artifact deposu.

    Önerilen klasör yapısı:

    data/
    ├── reports/
    │   └── arcelik/
    │       └── 2023/
    │           ├── pdf/
    │           ├── renders/
    │           ├── ocr/
    │           ├── layouts/
    │           ├── classifications/
    │           ├── tables/
    │           ├── chunks/
    │           ├── evidence/
    │           ├── answers/
    │           └── temp/
    ├── cache/
    ├── exports/
    └── logs/

    Dosya yazma işlemleri mümkün olduğunca atomik yapılır. Önce geçici
    dosyaya yazılır, ardından os.replace ile hedef dosyaya taşınır.
    """

    _INVALID_FILENAME_CHARS = re.compile(
        r'[<>:"/\\|?*\x00-\x1F]'
    )

    _WHITESPACE_RE = re.compile(r"\s+")
    _MULTIPLE_DASH_RE = re.compile(r"-{2,}")

    def __init__(
        self,
        root_dir: str | Path = "data",
        *,
        reports_dir_name: str = "reports",
        cache_dir_name: str = "cache",
        exports_dir_name: str = "exports",
        logs_dir_name: str = "logs",
        create_directories: bool = True,
    ) -> None:
        self.root_dir = Path(
            root_dir
        ).expanduser().resolve()

        self.reports_dir = (
            self.root_dir / reports_dir_name
        )
        self.cache_dir = (
            self.root_dir / cache_dir_name
        )
        self.exports_dir = (
            self.root_dir / exports_dir_name
        )
        self.logs_dir = (
            self.root_dir / logs_dir_name
        )

        if create_directories:
            self.initialize()

    def initialize(self) -> None:
        """Dosya deposunun temel klasörlerini oluşturur."""

        for directory in (
            self.root_dir,
            self.reports_dir,
            self.cache_dir,
            self.exports_dir,
            self.logs_dir,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    def get_report_paths(
        self,
        company_name: str,
        report_year: int,
        *,
        report_id: str | None = None,
        create: bool = True,
    ) -> ReportPaths:
        """
        Bir şirket ve yıl için standart rapor klasörlerini döndürür.

        Aynı şirket ve yıl için birden fazla rapor tutulacaksa report_id
        verilebilir.
        """

        company_slug = self.slugify(
            company_name
        )

        self._validate_year(
            report_year
        )

        report_root = (
            self.reports_dir
            / company_slug
            / str(report_year)
        )

        if report_id:
            report_root = (
                report_root
                / self.sanitize_filename(report_id)
            )

        paths = ReportPaths(
            report_root=report_root,
            pdf_dir=report_root / "pdf",
            render_dir=report_root / "renders",
            ocr_dir=report_root / "ocr",
            layout_dir=report_root / "layouts",
            classification_dir=(
                report_root / "classifications"
            ),
            table_dir=report_root / "tables",
            chunk_dir=report_root / "chunks",
            evidence_dir=report_root / "evidence",
            answer_dir=report_root / "answers",
            temp_dir=report_root / "temp",
        )

        if create:
            paths.create()

        return paths

    def save_pdf(
        self,
        source_path: str | Path,
        *,
        company_name: str,
        report_year: int,
        report_id: str | None = None,
        file_name: str | None = None,
        overwrite: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> StoredArtifact:
        """
        İndirilmiş bir PDF'yi standart rapor klasörüne kopyalar.

        Kaynak dosya korunur. Hedef dosya atomik olarak oluşturulur.
        """

        source = Path(
            source_path
        ).expanduser().resolve()

        if not source.is_file():
            raise FileNotFoundError(
                f"Kaynak PDF bulunamadı: {source}"
            )

        if source.suffix.lower() != ".pdf":
            raise ValueError(
                "save_pdf yalnızca .pdf dosyaları kabul eder."
            )

        paths = self.get_report_paths(
            company_name=company_name,
            report_year=report_year,
            report_id=report_id,
        )

        company_slug = self.slugify(
            company_name
        )

        target_name = (
            file_name
            or f"{company_slug}_{report_year}_sustainability_report.pdf"
        )

        target_name = self.ensure_extension(
            self.sanitize_filename(target_name),
            ".pdf",
        )

        target_path = (
            paths.pdf_dir / target_name
        )

        self.copy_file_atomic(
            source,
            target_path,
            overwrite=overwrite,
        )

        return self._build_artifact(
            artifact_type=ArtifactType.PDF,
            path=target_path,
            company_slug=company_slug,
            report_year=report_year,
            report_id=report_id,
            metadata=metadata,
        )

    def save_render(
        self,
        source_path: str | Path,
        *,
        company_name: str,
        report_year: int,
        page_number: int,
        report_id: str | None = None,
        overwrite: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> StoredArtifact:
        """Render edilmiş sayfa görüntüsünü standart konuma kopyalar."""

        if page_number < 1:
            raise ValueError(
                "page_number 1 veya daha büyük olmalıdır."
            )

        source = Path(
            source_path
        ).expanduser().resolve()

        if not source.is_file():
            raise FileNotFoundError(
                f"Render dosyası bulunamadı: {source}"
            )

        extension = source.suffix.lower()

        if extension not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            raise ValueError(
                "Desteklenmeyen render formatı: "
                f"{extension}"
            )

        paths = self.get_report_paths(
            company_name=company_name,
            report_year=report_year,
            report_id=report_id,
        )

        target_name = (
            f"page_{page_number:04d}{extension}"
        )

        target_path = (
            paths.render_dir / target_name
        )

        self.copy_file_atomic(
            source,
            target_path,
            overwrite=overwrite,
        )

        return self._build_artifact(
            artifact_type=ArtifactType.RENDER,
            path=target_path,
            company_slug=self.slugify(company_name),
            report_year=report_year,
            report_id=report_id,
            page_number=page_number,
            metadata=metadata,
        )

    def save_json(
        self,
        data: Any,
        target_path: str | Path,
        *,
        overwrite: bool = True,
        indent: int = 2,
        metadata: Mapping[str, Any] | None = None,
        artifact_type: ArtifactType = ArtifactType.OTHER,
        company_slug: str | None = None,
        report_year: int | None = None,
        report_id: str | None = None,
        page_number: int | None = None,
    ) -> StoredArtifact:
        """Python nesnesini JSON dosyası olarak atomik biçimde kaydeder."""

        target = self._resolve_target_path(
            target_path
        )

        target = target.with_suffix(
            ".json"
        )

        payload = self._serialize_value(
            data
        )

        self.write_text_atomic(
            target,
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=indent,
                sort_keys=False,
            ),
            overwrite=overwrite,
        )

        return self._build_artifact(
            artifact_type=artifact_type,
            path=target,
            company_slug=company_slug,
            report_year=report_year,
            report_id=report_id,
            page_number=page_number,
            metadata=metadata,
        )

    def load_json(
        self,
        file_path: str | Path,
    ) -> Any:
        """Bir JSON dosyasını okuyup Python nesnesi olarak döndürür."""

        path = self._resolve_existing_path(
            file_path
        )

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)

        except json.JSONDecodeError as exc:
            raise FileStoreError(
                f"Geçersiz JSON dosyası: {path}"
            ) from exc

        except OSError as exc:
            raise FileStoreError(
                f"JSON dosyası okunamadı: {path}"
            ) from exc

    def save_text(
        self,
        text: str,
        target_path: str | Path,
        *,
        overwrite: bool = True,
        artifact_type: ArtifactType = ArtifactType.OTHER,
        company_slug: str | None = None,
        report_year: int | None = None,
        report_id: str | None = None,
        page_number: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> StoredArtifact:
        """Metin içeriğini UTF-8 olarak kaydeder."""

        target = self._resolve_target_path(
            target_path
        )

        self.write_text_atomic(
            target,
            text,
            overwrite=overwrite,
        )

        return self._build_artifact(
            artifact_type=artifact_type,
            path=target,
            company_slug=company_slug,
            report_year=report_year,
            report_id=report_id,
            page_number=page_number,
            metadata=metadata,
        )

    def load_text(
        self,
        file_path: str | Path,
    ) -> str:
        """UTF-8 metin dosyasını okur."""

        path = self._resolve_existing_path(
            file_path
        )

        try:
            return path.read_text(
                encoding="utf-8"
            )

        except OSError as exc:
            raise FileStoreError(
                f"Metin dosyası okunamadı: {path}"
            ) from exc

    def save_bytes(
        self,
        data: bytes,
        target_path: str | Path,
        *,
        overwrite: bool = True,
        artifact_type: ArtifactType = ArtifactType.OTHER,
        company_slug: str | None = None,
        report_year: int | None = None,
        report_id: str | None = None,
        page_number: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> StoredArtifact:
        """Binary veriyi atomik biçimde dosyaya yazar."""

        target = self._resolve_target_path(
            target_path
        )

        self.write_bytes_atomic(
            target,
            data,
            overwrite=overwrite,
        )

        return self._build_artifact(
            artifact_type=artifact_type,
            path=target,
            company_slug=company_slug,
            report_year=report_year,
            report_id=report_id,
            page_number=page_number,
            metadata=metadata,
        )

    def create_manifest(
        self,
        *,
        company_name: str,
        report_year: int,
        report_id: str | None = None,
        artifacts: Sequence[StoredArtifact],
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> StoredArtifact:
        """Bir rapora ait artifact listesini manifest dosyasına yazar."""

        paths = self.get_report_paths(
            company_name=company_name,
            report_year=report_year,
            report_id=report_id,
        )

        payload = {
            "company_name": company_name,
            "company_slug": self.slugify(
                company_name
            ),
            "report_year": report_year,
            "report_id": report_id,
            "created_at": datetime.now(
                UTC
            ).isoformat(),
            "artifact_count": len(artifacts),
            "artifacts": [
                artifact.to_dict()
                for artifact in artifacts
            ],
            "metadata": dict(
                extra_metadata or {}
            ),
        }

        return self.save_json(
            payload,
            paths.report_root / "manifest.json",
            overwrite=True,
            artifact_type=ArtifactType.OTHER,
            company_slug=self.slugify(company_name),
            report_year=report_year,
            report_id=report_id,
        )

    def list_artifacts(
        self,
        *,
        company_name: str | None = None,
        report_year: int | None = None,
        report_id: str | None = None,
        artifact_type: ArtifactType | None = None,
        recursive: bool = True,
    ) -> list[Path]:
        """Belirtilen filtrelerle depodaki dosyaları listeler."""

        if company_name is None:
            base_dir = self.root_dir

        elif report_year is None:
            base_dir = (
                self.reports_dir
                / self.slugify(company_name)
            )

        else:
            base_dir = self.get_report_paths(
                company_name=company_name,
                report_year=report_year,
                report_id=report_id,
                create=False,
            ).report_root

        if artifact_type is not None:
            base_dir = self._artifact_directory(
                base_dir=base_dir,
                artifact_type=artifact_type,
            )

        if not base_dir.exists():
            return []

        iterator = (
            base_dir.rglob("*")
            if recursive
            else base_dir.glob("*")
        )

        return sorted(
            path
            for path in iterator
            if path.is_file()
        )

    def exists(
        self,
        file_path: str | Path,
    ) -> bool:
        """Dosyanın depo içinde var olup olmadığını kontrol eder."""

        try:
            path = self._resolve_target_path(
                file_path
            )
        except FileStoreError:
            return False

        return path.is_file()

    def delete_file(
        self,
        file_path: str | Path,
        *,
        missing_ok: bool = True,
    ) -> None:
        """Depo içindeki tek bir dosyayı siler."""

        path = self._resolve_target_path(
            file_path
        )

        if not path.exists():
            if missing_ok:
                return

            raise FileNotFoundError(
                f"Silinecek dosya bulunamadı: {path}"
            )

        if not path.is_file():
            raise FileStoreError(
                f"Hedef bir dosya değil: {path}"
            )

        try:
            path.unlink()

        except OSError as exc:
            raise FileStoreError(
                f"Dosya silinemedi: {path}"
            ) from exc

    def clear_report(
        self,
        *,
        company_name: str,
        report_year: int,
        report_id: str | None = None,
        confirm: bool = False,
    ) -> None:
        """Bir rapora ait bütün artifact klasörünü siler."""

        if not confirm:
            raise FileStoreError(
                "Rapor klasörünü silmek için confirm=True verilmelidir."
            )

        paths = self.get_report_paths(
            company_name=company_name,
            report_year=report_year,
            report_id=report_id,
            create=False,
        )

        if not paths.report_root.exists():
            return

        try:
            shutil.rmtree(
                paths.report_root
            )

        except OSError as exc:
            raise FileStoreError(
                "Rapor klasörü silinemedi: "
                f"{paths.report_root}"
            ) from exc

    def copy_file_atomic(
        self,
        source_path: str | Path,
        target_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Bir dosyayı hedef konuma atomik biçimde kopyalar."""

        source = Path(
            source_path
        ).expanduser().resolve()

        target = self._resolve_target_path(
            target_path
        )

        if not source.is_file():
            raise FileNotFoundError(
                f"Kaynak dosya bulunamadı: {source}"
            )

        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Hedef dosya zaten mevcut: {target}"
            )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".part",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(
                    temporary_file.name
                )

                with source.open("rb") as source_file:
                    shutil.copyfileobj(
                        source_file,
                        temporary_file,
                        length=1024 * 1024,
                    )

                temporary_file.flush()
                os.fsync(
                    temporary_file.fileno()
                )

            os.replace(
                temporary_path,
                target,
            )

            return target

        except OSError as exc:
            raise FileStoreError(
                f"Dosya kopyalanamadı: {source} -> {target}"
            ) from exc

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink(
                    missing_ok=True
                )

    def write_text_atomic(
        self,
        target_path: str | Path,
        text: str,
        *,
        overwrite: bool = True,
    ) -> Path:
        """Metin dosyasını atomik biçimde yazar."""

        target = self._resolve_target_path(
            target_path
        )

        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Hedef dosya zaten mevcut: {target}"
            )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".part",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(
                    temporary_file.name
                )

                temporary_file.write(text)
                temporary_file.flush()
                os.fsync(
                    temporary_file.fileno()
                )

            os.replace(
                temporary_path,
                target,
            )

            return target

        except OSError as exc:
            raise FileStoreError(
                f"Metin dosyası yazılamadı: {target}"
            ) from exc

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink(
                    missing_ok=True
                )

    def write_bytes_atomic(
        self,
        target_path: str | Path,
        data: bytes,
        *,
        overwrite: bool = True,
    ) -> Path:
        """Binary dosyayı atomik biçimde yazar."""

        target = self._resolve_target_path(
            target_path
        )

        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Hedef dosya zaten mevcut: {target}"
            )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".part",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(
                    temporary_file.name
                )

                temporary_file.write(data)
                temporary_file.flush()
                os.fsync(
                    temporary_file.fileno()
                )

            os.replace(
                temporary_path,
                target,
            )

            return target

        except OSError as exc:
            raise FileStoreError(
                f"Binary dosya yazılamadı: {target}"
            ) from exc

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink(
                    missing_ok=True
                )

    def calculate_sha256(
        self,
        file_path: str | Path,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """Dosyanın SHA-256 özetini hesaplar."""

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size pozitif olmalıdır."
            )

        path = self._resolve_existing_path(
            file_path
        )

        digest = hashlib.sha256()

        try:
            with path.open("rb") as file:
                while chunk := file.read(
                    chunk_size
                ):
                    digest.update(chunk)

        except OSError as exc:
            raise FileStoreError(
                f"Dosya hash'i hesaplanamadı: {path}"
            ) from exc

        return digest.hexdigest()

    def relative_to_root(
        self,
        file_path: str | Path,
    ) -> Path:
        """Bir dosyanın depo köküne göre bağıl yolunu döndürür."""

        path = Path(
            file_path
        ).expanduser().resolve()

        try:
            return path.relative_to(
                self.root_dir
            )

        except ValueError as exc:
            raise FileStoreError(
                "Dosya depo kökünün dışında: "
                f"{path}"
            ) from exc

    @classmethod
    def slugify(
        cls,
        value: str,
    ) -> str:
        """Şirket adını dosya sistemiyle uyumlu slug biçimine dönüştürür."""

        if not value or not value.strip():
            raise ValueError(
                "Slug üretmek için boş değer kullanılamaz."
            )

        translation_table = str.maketrans(
            {
                "ç": "c",
                "Ç": "c",
                "ğ": "g",
                "Ğ": "g",
                "ı": "i",
                "I": "i",
                "İ": "i",
                "ö": "o",
                "Ö": "o",
                "ş": "s",
                "Ş": "s",
                "ü": "u",
                "Ü": "u",
            }
        )

        normalized = value.translate(
            translation_table
        ).lower()

        normalized = cls._WHITESPACE_RE.sub(
            "-",
            normalized.strip(),
        )

        normalized = re.sub(
            r"[^a-z0-9._-]",
            "-",
            normalized,
        )

        normalized = cls._MULTIPLE_DASH_RE.sub(
            "-",
            normalized,
        )

        normalized = normalized.strip(
            "._-"
        )

        if not normalized:
            raise ValueError(
                "Geçerli bir slug üretilemedi."
            )

        return normalized

    @classmethod
    def sanitize_filename(
        cls,
        file_name: str,
        *,
        replacement: str = "_",
        max_length: int = 180,
    ) -> str:
        """Dosya adındaki geçersiz karakterleri temizler."""

        if not file_name or not file_name.strip():
            raise ValueError(
                "Dosya adı boş olamaz."
            )

        sanitized = cls._INVALID_FILENAME_CHARS.sub(
            replacement,
            file_name.strip(),
        )

        sanitized = cls._WHITESPACE_RE.sub(
            "_",
            sanitized,
        )

        sanitized = sanitized.rstrip(
            ". "
        )

        if len(sanitized) > max_length:
            path = Path(sanitized)

            extension = path.suffix

            stem_length = max(
                max_length - len(extension),
                1,
            )

            sanitized = (
                path.stem[:stem_length]
                + extension
            )

        if sanitized in {
            "",
            ".",
            "..",
        }:
            raise ValueError(
                "Geçerli bir dosya adı üretilemedi."
            )

        return sanitized

    @staticmethod
    def ensure_extension(
        file_name: str,
        extension: str,
    ) -> str:
        """Dosya adının belirtilen uzantıyla bitmesini sağlar."""

        normalized_extension = (
            extension
            if extension.startswith(".")
            else f".{extension}"
        )

        if file_name.lower().endswith(
            normalized_extension.lower()
        ):
            return file_name

        return (
            f"{file_name}{normalized_extension}"
        )

    def _build_artifact(
        self,
        *,
        artifact_type: ArtifactType,
        path: Path,
        company_slug: str | None = None,
        report_year: int | None = None,
        report_id: str | None = None,
        page_number: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> StoredArtifact:
        path = path.resolve()

        if not path.is_file():
            raise FileStoreError(
                f"Artifact dosyası bulunamadı: {path}"
            )

        metadata_path: Path | None = None

        if metadata:
            metadata_path = path.with_suffix(
                f"{path.suffix}.metadata.json"
            )

            metadata_payload = {
                "artifact_type": artifact_type.value,
                "artifact_path": str(
                    self.relative_to_root(path)
                ),
                "company_slug": company_slug,
                "report_year": report_year,
                "report_id": report_id,
                "page_number": page_number,
                "created_at": datetime.now(
                    UTC
                ).isoformat(),
                "metadata": self._serialize_value(
                    metadata
                ),
            }

            self.write_text_atomic(
                metadata_path,
                json.dumps(
                    metadata_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                overwrite=True,
            )

        stat = path.stat()

        return StoredArtifact(
            artifact_type=artifact_type,
            path=path,
            relative_path=self.relative_to_root(
                path
            ),
            file_name=path.name,
            file_extension=path.suffix.lower(),
            file_size_bytes=stat.st_size,
            sha256=self.calculate_sha256(
                path
            ),
            created_at=datetime.fromtimestamp(
                stat.st_ctime,
                tz=UTC,
            ),
            company_slug=company_slug,
            report_year=report_year,
            report_id=report_id,
            page_number=page_number,
            metadata_path=metadata_path,
        )

    def _resolve_target_path(
        self,
        path: str | Path,
    ) -> Path:
        candidate = Path(
            path
        ).expanduser()

        if not candidate.is_absolute():
            candidate = (
                self.root_dir / candidate
            )

        candidate = candidate.resolve()

        try:
            candidate.relative_to(
                self.root_dir
            )

        except ValueError as exc:
            raise FileStoreError(
                "Hedef yol depo kökünün dışında olamaz: "
                f"{candidate}"
            ) from exc

        return candidate

    def _resolve_existing_path(
        self,
        path: str | Path,
    ) -> Path:
        candidate = self._resolve_target_path(
            path
        )

        if not candidate.is_file():
            raise FileNotFoundError(
                f"Dosya bulunamadı: {candidate}"
            )

        return candidate

    def _artifact_directory(
        self,
        *,
        base_dir: Path,
        artifact_type: ArtifactType,
    ) -> Path:
        directory_map = {
            ArtifactType.PDF: "pdf",
            ArtifactType.RENDER: "renders",
            ArtifactType.OCR: "ocr",
            ArtifactType.LAYOUT: "layouts",
            ArtifactType.CLASSIFICATION: (
                "classifications"
            ),
            ArtifactType.TABLE: "tables",
            ArtifactType.CHUNK: "chunks",
            ArtifactType.EVIDENCE: "evidence",
            ArtifactType.ANSWER: "answers",
        }

        directory_name = directory_map.get(
            artifact_type
        )

        if directory_name is None:
            return base_dir

        return base_dir / directory_name

    @staticmethod
    def _serialize_value(
        value: Any,
    ) -> Any:
        """Dataclass, Path, Enum ve datetime nesnelerini JSON uyumlu yapar."""

        if is_dataclass(value):
            return FileStore._serialize_value(
                asdict(value)
            )

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, StrEnum):
            return value.value

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Mapping):
            return {
                str(key): FileStore._serialize_value(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            Sequence,
        ) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return [
                FileStore._serialize_value(
                    item
                )
                for item in value
            ]

        if isinstance(value, set):
            return [
                FileStore._serialize_value(
                    item
                )
                for item in sorted(
                    value,
                    key=str,
                )
            ]

        return value

    @staticmethod
    def _validate_year(
        year: int,
    ) -> None:
        current_year = datetime.now(
            UTC
        ).year

        if not 1900 <= year <= current_year + 1:
            raise ValueError(
                f"Geçersiz rapor yılı: {year}"
            )


def create_file_store(
    root_dir: str | Path = "data",
) -> FileStore:
    """FileStore için kısa oluşturma fonksiyonu."""

    return FileStore(
        root_dir=root_dir
    )


__all__ = [
    "ArtifactType",
    "FileStore",
    "FileStoreError",
    "ReportPaths",
    "StoredArtifact",
    "create_file_store",
]
