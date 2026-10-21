from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping

from cloud_sync.report_input_restore import ReportInputRestore, RestoreResult
from cloud_sync.report_output_sync import ReportOutputSync, SyncResult
from storage_backends import create_storage_backend


def sync_report_outputs(
    *,
    report_id: str,
    data_directory: Path | str,
    remote_prefix: str | None = None,
    overwrite: bool = True,
    additional_files: Mapping[str, Path | str] | None = None,
    include_sqlite_state: bool = True,
    include_global_vectors: bool = False,
    include_report_directories: bool = True,
) -> SyncResult:
    storage_backend = create_storage_backend()
    sync_service = ReportOutputSync(
        storage_backend=storage_backend,
        data_directory=data_directory,
    )
    return sync_service.sync_report(
        report_id=report_id,
        remote_prefix=remote_prefix,
        overwrite=overwrite,
        additional_files=additional_files,
        include_sqlite_state=include_sqlite_state,
        include_global_vectors=include_global_vectors,
        include_report_directories=include_report_directories,
    )


def restore_report_inputs(
    *,
    report_id: str,
    data_directory: Path | str,
    remote_prefix: str | None = None,
    overwrite: bool = True,
    require_database: bool = True,
    include_global_vectors: bool = False,
    include_report_assets: bool = False,
) -> RestoreResult:
    storage_backend = create_storage_backend()
    restore_service = ReportInputRestore(
        storage_backend=storage_backend,
        data_directory=data_directory,
    )
    return restore_service.restore_report(
        report_id=report_id,
        remote_prefix=remote_prefix,
        overwrite=overwrite,
        require_database=require_database,
        include_global_vectors=include_global_vectors,
        include_report_assets=include_report_assets,
    )


def print_sync_result(result: SyncResult) -> None:
    size_mb = result.uploaded_byte_count / (1024 * 1024)
    print()
    print("Cloud senkronizasyonu tamamlandı.")
    print(f"  Report ID: {result.report_id}")
    print(f"  Hedef prefix: {result.remote_prefix}")
    print(f"  Dosya sayısı: {result.uploaded_file_count}")
    print(f"  Toplam boyut: {size_mb:.2f} MB")
    print(f"  Tamamlanma zamanı: {result.completed_at}")


def print_restore_result(result: RestoreResult) -> None:
    size_mb = result.downloaded_byte_count / (1024 * 1024)
    print()
    print("Cloud restore tamamlandı.")
    print(f"  Report ID: {result.report_id}")
    print(f"  Kaynak prefix: {result.remote_prefix}")
    print(f"  Dosya sayısı: {result.downloaded_file_count}")
    print(f"  Toplam boyut: {size_mb:.2f} MB")
    print(f"  SQLite restore: {result.restored_database}")
    print(f"  Tamamlanma zamanı: {result.completed_at}")


def print_sync_error(exc: Exception) -> None:
    print(f"Cloud senkronizasyonu başarısız: {exc}", file=sys.stderr)


def print_restore_error(exc: Exception) -> None:
    print(f"Cloud restore başarısız: {exc}", file=sys.stderr)


def sync_result_as_json(result: SyncResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


def restore_result_as_json(result: RestoreResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)