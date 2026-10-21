from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from .base import StorageBackend


class LocalStorageBackend(StorageBackend):
    def __init__(self, root_directory: Path | str) -> None:
        self.root_directory = Path(root_directory).resolve()
        self.root_directory.mkdir(parents=True, exist_ok=True)

    def upload_file(
        self,
        local_path: Path,
        remote_path: str,
        *,
        overwrite: bool = True,
    ) -> None:
        source = Path(local_path)
        destination = self._resolve_remote_path(remote_path)

        if not source.is_file():
            raise FileNotFoundError(f"Yüklenecek dosya bulunamadı: {source}")

        if destination.exists() and not overwrite:
            raise FileExistsError(f"Hedef dosya zaten mevcut: {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def download_file(
        self,
        remote_path: str,
        local_path: Path,
        *,
        overwrite: bool = True,
    ) -> Path:
        source = self._resolve_remote_path(remote_path)
        destination = Path(local_path)

        if not source.is_file():
            raise FileNotFoundError(f"Kaynak dosya bulunamadı: {source}")

        if destination.exists() and not overwrite:
            raise FileExistsError(f"Hedef dosya zaten mevcut: {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def exists(self, remote_path: str) -> bool:
        return self._resolve_remote_path(remote_path).is_file()

    def list_files(self, prefix: str = "") -> Iterable[str]:
        base_path = self._resolve_remote_path(prefix)

        if base_path.is_file():
            return [base_path.relative_to(self.root_directory).as_posix()]

        if not base_path.exists():
            return []

        return [
            path.relative_to(self.root_directory).as_posix()
            for path in base_path.rglob("*")
            if path.is_file()
        ]

    def _resolve_remote_path(self, remote_path: str) -> Path:
        normalized = remote_path.replace("\\", "/").lstrip("/")
        candidate = (self.root_directory / normalized).resolve()

        if self.root_directory not in candidate.parents and candidate != self.root_directory:
            raise ValueError(f"Geçersiz uzak dosya yolu: {remote_path}")

        return candidate