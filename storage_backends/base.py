from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

UploadProgressCallback = Callable[[Path, str, int, int], None]
DownloadProgressCallback = Callable[[str, Path, int, int], None]


class StorageBackend(ABC):
    @abstractmethod
    def upload_file(
        self,
        local_path: Path,
        remote_path: str,
        *,
        overwrite: bool = True,
    ) -> None:
        """Yerel dosyayı depolama alanına yükler."""

    @abstractmethod
    def download_file(
        self,
        remote_path: str,
        local_path: Path,
        *,
        overwrite: bool = True,
    ) -> Path:
        """Depolama alanındaki dosyayı yerel sisteme indirir."""

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Dosyanın depolama alanında bulunup bulunmadığını döndürür."""

    @abstractmethod
    def list_files(self, prefix: str = "") -> Iterable[str]:
        """Belirtilen prefix altındaki dosyaları listeler."""

    def upload_directory(
        self,
        local_directory: Path,
        remote_prefix: str,
        *,
        overwrite: bool = True,
        progress_callback: UploadProgressCallback | None = None,
    ) -> list[str]:
        """Klasör altındaki dosyaları göreli yollarını koruyarak yükler."""
        directory = Path(local_directory)

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Yüklenecek klasör bulunamadı: {directory}"
            )

        local_files = sorted(
            path for path in directory.rglob("*") if path.is_file()
        )
        total_count = len(local_files)
        uploaded_files: list[str] = []

        for completed_count, local_file in enumerate(local_files, start=1):
            relative_path = local_file.relative_to(directory).as_posix()
            remote_path = self._join_remote_path(
                remote_prefix,
                relative_path,
            )

            self.upload_file(
                local_file,
                remote_path,
                overwrite=overwrite,
            )
            uploaded_files.append(remote_path)

            if progress_callback is not None:
                progress_callback(
                    local_file,
                    remote_path,
                    completed_count,
                    total_count,
                )

        return uploaded_files

    def download_directory(
        self,
        remote_prefix: str,
        local_directory: Path,
        *,
        overwrite: bool = True,
        progress_callback: DownloadProgressCallback | None = None,
    ) -> list[Path]:
        """
        Uzak prefix altındaki dosyaları yerel klasöre indirir.

        Blob adları normalize edilir ve path traversal girişimleri reddedilir.
        """
        normalized_prefix = remote_prefix.replace("\\", "/").strip("/")
        prefix_with_separator = (
            f"{normalized_prefix}/" if normalized_prefix else ""
        )
        destination_root = Path(local_directory).resolve()
        destination_root.mkdir(parents=True, exist_ok=True)

        remote_files = sorted(self.list_files(prefix_with_separator))
        downloaded_files: list[Path] = []
        total_count = len(remote_files)

        for completed_count, remote_path in enumerate(remote_files, start=1):
            normalized_remote = remote_path.replace("\\", "/").strip("/")

            if prefix_with_separator:
                if not normalized_remote.startswith(prefix_with_separator):
                    continue
                relative_text = normalized_remote[len(prefix_with_separator):]
            else:
                relative_text = normalized_remote

            relative_path = PurePosixPath(relative_text)
            if (
                not relative_text
                or relative_path.is_absolute()
                or ".." in relative_path.parts
            ):
                raise ValueError(f"Geçersiz uzak dosya yolu: {remote_path}")

            destination = destination_root.joinpath(*relative_path.parts)
            resolved_destination = destination.resolve()
            if destination_root not in resolved_destination.parents:
                raise ValueError(f"Hedef klasör dışına çıkan yol: {remote_path}")

            downloaded = self.download_file(
                normalized_remote,
                resolved_destination,
                overwrite=overwrite,
            )
            downloaded_files.append(downloaded)

            if progress_callback is not None:
                progress_callback(
                    normalized_remote,
                    downloaded,
                    completed_count,
                    total_count,
                )

        return downloaded_files

    @staticmethod
    def _join_remote_path(prefix: str, relative_path: str) -> str:
        normalized_prefix = prefix.replace("\\", "/").strip("/")
        normalized_relative = relative_path.replace("\\", "/").strip("/")

        if normalized_prefix:
            return f"{normalized_prefix}/{normalized_relative}"

        return normalized_relative