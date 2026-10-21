from __future__ import annotations

import os
from pathlib import Path

from .azure_blob_storage import AzureBlobStorageBackend
from .base import StorageBackend
from .local_storage import LocalStorageBackend


class StorageBackendConfigurationError(RuntimeError):
    """Storage backend ayarları eksik veya geçersiz olduğunda oluşur."""


def create_storage_backend(
    *,
    backend_name: str | None = None,
    local_root_directory: Path | str | None = None,
) -> StorageBackend:
    """
    Ortam değişkenlerine göre uygun storage backend nesnesini oluşturur.

    Desteklenen backend değerleri:
    - local
    - azure_blob
    """

    selected_backend = (
        backend_name
        or os.getenv("STORAGE_BACKEND", "local")
    ).strip().lower()

    if selected_backend == "local":
        root_directory = (
            local_root_directory
            or os.getenv("LOCAL_STORAGE_ROOT")
            or "data/cloud_storage"
        )

        return LocalStorageBackend(
            root_directory=Path(root_directory)
        )

    if selected_backend == "azure_blob":
        connection_string = os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING",
            "",
        ).strip()

        container_name = os.getenv(
            "AZURE_STORAGE_CONTAINER",
            "",
        ).strip()

        if not connection_string:
            raise StorageBackendConfigurationError(
                "STORAGE_BACKEND=azure_blob seçildi ancak "
                "AZURE_STORAGE_CONNECTION_STRING tanımlı değil."
            )

        if not container_name:
            raise StorageBackendConfigurationError(
                "STORAGE_BACKEND=azure_blob seçildi ancak "
                "AZURE_STORAGE_CONTAINER tanımlı değil."
            )

        return AzureBlobStorageBackend(
            connection_string=connection_string,
            container_name=container_name,
        )

    raise StorageBackendConfigurationError(
        "Desteklenmeyen STORAGE_BACKEND değeri: "
        f"{selected_backend!r}. Desteklenen değerler: "
        "'local', 'azure_blob'."
    )