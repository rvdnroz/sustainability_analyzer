from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from storage_backends import StorageBackend


@dataclass(frozen=True)
class SyncResult:
    report_id: str
    remote_prefix: str
    uploaded_file_count: int
    uploaded_byte_count: int
    uploaded_files: list[str]
    completed_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ReportOutputSync:
    """
    Bir rapora ait yerel çıktıları ve gerekli kalıcı durum dosyalarını
    uzak depolamaya yükler.

    Rapor bazlı klasörler:
        data/files/<report_id>
        data/table_images/<report_id>
        data/pages/<report_id>
        data/rendered_pages/<report_id>
        data/tables/<report_id>
        data/chunks/<report_id>
        data/vectors/<report_id>
        data/outputs/<report_id>

    İsteğe bağlı ek kapsam:
        - Proje kökündeki outputs/*.xlsx benzeri ek dosyalar
        - data/sustainability.db güvenli SQLite yedeği
        - data/vectors altındaki global vector-store durumu
    """

    DEFAULT_REPORT_DIRECTORIES = (
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
        self.progress_callback = progress_callback

        if not self.data_directory.is_dir():
            raise NotADirectoryError(
                f"Data klasörü bulunamadı: {self.data_directory}"
            )

    def sync_report(
        self,
        *,
        report_id: str,
        remote_prefix: str | None = None,
        overwrite: bool = True,
        additional_files: Mapping[str, Path | str] | None = None,
        include_sqlite_state: bool = True,
        include_global_vectors: bool = False,
        include_report_directories: bool = True,
    ) -> SyncResult:
        normalized_report_id = report_id.strip()

        if not normalized_report_id:
            raise ValueError("report_id boş olamaz.")

        destination_prefix = (
            remote_prefix.strip("/")
            if remote_prefix
            else f"reports/{normalized_report_id}"
        )

        uploaded_files: list[str] = []
        uploaded_byte_count = 0

        self._emit(
            f"Cloud senkronizasyonu başlıyor: {destination_prefix}"
        )

        if include_report_directories:
            for directory_name in self.DEFAULT_REPORT_DIRECTORIES:
                local_report_directory = (
                    self.data_directory
                    / directory_name
                    / normalized_report_id
                )

                if not local_report_directory.is_dir():
                    continue

                remote_directory_prefix = (
                    f"{destination_prefix}/{directory_name}"
                )

                self._emit(
                    f"Yükleniyor: {local_report_directory} "
                    f"-> {remote_directory_prefix}"
                )

                directory_files = self.storage_backend.upload_directory(
                    local_report_directory,
                    remote_directory_prefix,
                    overwrite=overwrite,
                    progress_callback=self._directory_progress,
                )

                uploaded_files.extend(directory_files)
                uploaded_byte_count += self._directory_size(
                    local_report_directory
                )
        else:
            self._emit(
                "Rapor klasörleri senkronizasyon dışında bırakıldı; "
                "yalnızca yeni çıktılar ve seçili durum dosyaları yüklenecek."
            )

        if additional_files:
            for remote_relative_path, local_path_value in additional_files.items():
                local_path = Path(local_path_value).resolve()

                if not local_path.is_file():
                    raise FileNotFoundError(
                        f"Ek senkronizasyon dosyası bulunamadı: {local_path}"
                    )

                remote_path = self.storage_backend._join_remote_path(
                    destination_prefix,
                    remote_relative_path,
                )

                self._emit(f"Yükleniyor: {local_path} -> {remote_path}")
                self.storage_backend.upload_file(
                    local_path,
                    remote_path,
                    overwrite=overwrite,
                )

                uploaded_files.append(remote_path)
                uploaded_byte_count += local_path.stat().st_size

        if include_sqlite_state:
            database_path = self.data_directory / "sustainability.db"

            if database_path.is_file():
                # Windows'ta Azure SDK dosya akışını kısa süre daha açık
                # tutabildiğinden TemporaryDirectory çıkışında WinError 32
                # oluşabiliyor. Snapshot'ı kalıcı bir çalışma klasöründe
                # tutarak anlık silme yarışını ortadan kaldırıyoruz.
                snapshot_path = (
                    self.data_directory
                    / "sync_snapshots"
                    / normalized_report_id
                    / "sustainability.db"
                )
                self._create_sqlite_snapshot(
                    source_path=database_path,
                    destination_path=snapshot_path,
                )

                remote_path = (
                    f"{destination_prefix}/state/sustainability.db"
                )

                self._emit(
                    f"SQLite yedeği yükleniyor: {remote_path}"
                )
                self.storage_backend.upload_file(
                    snapshot_path,
                    remote_path,
                    overwrite=overwrite,
                )

                uploaded_files.append(remote_path)
                uploaded_byte_count += snapshot_path.stat().st_size

        if include_global_vectors:
            global_vectors_directory = self.data_directory / "vectors"

            if global_vectors_directory.is_dir():
                remote_vector_prefix = (
                    f"{destination_prefix}/state/vectors"
                )

                self._emit(
                    f"Global vector store yükleniyor: "
                    f"{global_vectors_directory} -> {remote_vector_prefix}"
                )

                vector_files = self.storage_backend.upload_directory(
                    global_vectors_directory,
                    remote_vector_prefix,
                    overwrite=overwrite,
                    progress_callback=self._directory_progress,
                )

                uploaded_files.extend(vector_files)
                uploaded_byte_count += self._directory_size(
                    global_vectors_directory
                )

        if not uploaded_files:
            raise FileNotFoundError(
                "Yüklenecek rapor çıktısı veya durum dosyası bulunamadı. "
                f"report_id={normalized_report_id}, "
                f"data_directory={self.data_directory}"
            )

        completed_at = datetime.now(timezone.utc).isoformat()

        result = SyncResult(
            report_id=normalized_report_id,
            remote_prefix=destination_prefix,
            uploaded_file_count=len(uploaded_files),
            uploaded_byte_count=uploaded_byte_count,
            uploaded_files=uploaded_files,
            completed_at=completed_at,
        )

        self._upload_sync_manifest(
            result=result,
            overwrite=overwrite,
        )

        self._emit(
            f"Cloud senkronizasyonu tamamlandı: "
            f"{result.uploaded_file_count} dosya"
        )

        return result

    def _upload_sync_manifest(
        self,
        *,
        result: SyncResult,
        overwrite: bool,
    ) -> None:
        manifest_directory = (
            self.data_directory
            / "sync_manifests"
            / result.report_id
        )
        manifest_directory.mkdir(parents=True, exist_ok=True)

        manifest_path = manifest_directory / "blob_sync_manifest.json"

        manifest_path.write_text(
            json.dumps(
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.storage_backend.upload_file(
            manifest_path,
            f"{result.remote_prefix}/blob_sync_manifest.json",
            overwrite=overwrite,
        )

    @staticmethod
    def _create_sqlite_snapshot(
        *,
        source_path: Path,
        destination_path: Path,
    ) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        # Önceki snapshot varsa bağlantı açılmadan kaldırılır. Böylece
        # sqlite3 backup mevcut dosyanın üzerine güvenli biçimde yazılır.
        if destination_path.exists():
            destination_path.unlink()

        source_uri = f"file:{source_path.as_posix()}?mode=ro"

        source_connection = sqlite3.connect(source_uri, uri=True)
        destination_connection = sqlite3.connect(destination_path)

        try:
            source_connection.backup(destination_connection)
            destination_connection.commit()
        finally:
            destination_connection.close()
            source_connection.close()

    @staticmethod
    def _directory_size(directory: Path) -> int:
        return sum(
            file_path.stat().st_size
            for file_path in directory.rglob("*")
            if file_path.is_file()
        )

    def _directory_progress(
        self,
        local_file: Path,
        remote_path: str,
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
                f"{local_file.name} -> {remote_path}"
            )

    def _emit(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)