from __future__ import annotations

from pathlib import Path
from typing import Iterable

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

from .base import StorageBackend


class AzureBlobStorageBackend(StorageBackend):
    def __init__(
        self,
        *,
        connection_string: str,
        container_name: str,
    ) -> None:
        if not connection_string.strip():
            raise ValueError("Azure Storage connection string boş olamaz.")

        if not container_name.strip():
            raise ValueError("Azure Blob container adı boş olamaz.")

        self.container_name = container_name
        self.service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        self.container_client = self.service_client.get_container_client(
            container_name
        )

    def upload_file(
        self,
        local_path: Path,
        remote_path: str,
        *,
        overwrite: bool = True,
    ) -> None:
        source = Path(local_path)

        if not source.is_file():
            raise FileNotFoundError(f"Yüklenecek dosya bulunamadı: {source}")

        blob_name = self._normalize_blob_name(remote_path)
        blob_client = self.container_client.get_blob_client(blob_name)

        with source.open("rb") as file:
            blob_client.upload_blob(file, overwrite=overwrite)

    def download_file(
        self,
        remote_path: str,
        local_path: Path,
        *,
        overwrite: bool = True,
    ) -> Path:
        destination = Path(local_path)

        if destination.exists() and not overwrite:
            raise FileExistsError(f"Hedef dosya zaten mevcut: {destination}")

        blob_name = self._normalize_blob_name(remote_path)
        blob_client = self.container_client.get_blob_client(blob_name)

        try:
            stream = blob_client.download_blob()
        except ResourceNotFoundError as exc:
            raise FileNotFoundError(
                f"Azure Blob bulunamadı: {blob_name}"
            ) from exc

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = destination.with_name(f".{destination.name}.part")

        try:
            with temporary_path.open("wb") as output_file:
                for chunk in stream.chunks():
                    output_file.write(chunk)
            temporary_path.replace(destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        return destination

    def exists(self, remote_path: str) -> bool:
        blob_name = self._normalize_blob_name(remote_path)
        blob_client = self.container_client.get_blob_client(blob_name)

        try:
            return bool(blob_client.exists())
        except ResourceNotFoundError:
            return False

    def list_files(self, prefix: str = "") -> Iterable[str]:
        normalized_prefix = self._normalize_prefix(prefix)
        return [
            blob.name
            for blob in self.container_client.list_blobs(
                name_starts_with=normalized_prefix
            )
        ]

    @staticmethod
    def _normalize_blob_name(remote_path: str) -> str:
        normalized = remote_path.replace("\\", "/").strip("/")

        if not normalized:
            raise ValueError("Blob adı boş olamaz.")

        if ".." in normalized.split("/"):
            raise ValueError(f"Geçersiz blob yolu: {remote_path}")

        return normalized

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        return prefix.replace("\\", "/").lstrip("/")