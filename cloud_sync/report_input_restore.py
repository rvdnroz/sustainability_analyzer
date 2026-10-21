from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from storage_backends import StorageBackend


@dataclass(frozen=True)
class RestoreResult:
    report_id: str
    remote_prefix: str
    downloaded_file_count: int
    downloaded_byte_count: int
    downloaded_files: list[str]
    restored_database: bool
    completed_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ReportInputRestore:
    """Blob'daki rapor durumunu yeni container'ın yerel diskine geri yükler."""

    MINIMAL_REPORT_DIRECTORIES = (
        "files",
    )

    FULL_REPORT_DIRECTORIES = (
        "files",
        "table_images",
        "pages",
        "rendered_pages",
        "tables",
        "chunks",
        "vectors",
        "outputs",
    )

    def __init__(
        self,
        *,
        storage_backend: StorageBackend,
        data_directory: Path | str,
        progress_callback: Callable[[str], None] | None = print,
    ) -> None:
        self.storage_backend = storage_backend
        self.data_directory = Path(data_directory).resolve()
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback

    def restore_report(
        self,
        *,
        report_id: str,
        remote_prefix: str | None = None,
        overwrite: bool = True,
        require_database: bool = True,
        include_global_vectors: bool = False,
        include_report_assets: bool = False,
    ) -> RestoreResult:
        normalized_report_id = report_id.strip()
        if not normalized_report_id:
            raise ValueError("report_id boş olamaz.")

        source_prefix = (
            remote_prefix.strip("/")
            if remote_prefix
            else f"reports/{normalized_report_id}"
        )

        downloaded_paths: list[Path] = []
        restored_database = False

        self._emit(f"Cloud restore başlıyor: {source_prefix}")

        database_remote_path = f"{source_prefix}/state/sustainability.db"
        database_local_path = self.data_directory / "sustainability.db"

        if self.storage_backend.exists(database_remote_path):
            self._emit(
                f"SQLite durumu indiriliyor: {database_remote_path}"
            )
            downloaded_paths.append(
                self.storage_backend.download_file(
                    database_remote_path,
                    database_local_path,
                    overwrite=overwrite,
                )
            )
            restored_database = True
        elif require_database:
            raise FileNotFoundError(
                "Restore için gerekli SQLite veritabanı Blob'da bulunamadı: "
                f"{database_remote_path}"
            )

        report_directories = (
            self.FULL_REPORT_DIRECTORIES
            if include_report_assets
            else self.MINIMAL_REPORT_DIRECTORIES
        )

        if include_report_assets:
            self._emit(
                "Tam restore modu: rapor görselleri ve türetilmiş dosyalar "
                "indirilecek."
            )
        else:
            self._emit(
                "Minimal restore modu: yalnızca SQLite ve rapor metadata "
                "dosyaları indirilecek."
            )

        for directory_name in report_directories:
            remote_directory_prefix = f"{source_prefix}/{directory_name}"
            local_report_directory = (
                self.data_directory / directory_name / normalized_report_id
            )

            remote_files = list(
                self.storage_backend.list_files(
                    f"{remote_directory_prefix}/"
                )
            )
            if not remote_files:
                continue

            self._emit(
                f"İndiriliyor: {remote_directory_prefix} "
                f"-> {local_report_directory}"
            )
            downloaded_paths.extend(
                self.storage_backend.download_directory(
                    remote_directory_prefix,
                    local_report_directory,
                    overwrite=overwrite,
                    progress_callback=self._directory_progress,
                )
            )

        if include_global_vectors:
            remote_vector_prefix = f"{source_prefix}/state/vectors"
            remote_vector_files = list(
                self.storage_backend.list_files(f"{remote_vector_prefix}/")
            )
            if remote_vector_files:
                downloaded_paths.extend(
                    self.storage_backend.download_directory(
                        remote_vector_prefix,
                        self.data_directory / "vectors",
                        overwrite=overwrite,
                        progress_callback=self._directory_progress,
                    )
                )

        if not downloaded_paths:
            raise FileNotFoundError(
                "Restore edilecek rapor verisi bulunamadı. "
                f"report_id={normalized_report_id}, prefix={source_prefix}"
            )

        downloaded_byte_count = sum(
            path.stat().st_size for path in downloaded_paths if path.is_file()
        )
        completed_at = datetime.now(timezone.utc).isoformat()

        result = RestoreResult(
            report_id=normalized_report_id,
            remote_prefix=source_prefix,
            downloaded_file_count=len(downloaded_paths),
            downloaded_byte_count=downloaded_byte_count,
            downloaded_files=[str(path) for path in downloaded_paths],
            restored_database=restored_database,
            completed_at=completed_at,
        )

        self._emit(
            "Cloud restore tamamlandı: "
            f"{result.downloaded_file_count} dosya"
        )
        return result

    def _directory_progress(
        self,
        remote_path: str,
        local_path: Path,
        completed_count: int,
        total_count: int,
    ) -> None:
        if (
            completed_count == 1
            or completed_count == total_count
            or completed_count % 25 == 0
        ):
            self._emit(
                f"  [{completed_count}/{total_count}] "
                f"{remote_path} -> {local_path.name}"
            )

    def _emit(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)