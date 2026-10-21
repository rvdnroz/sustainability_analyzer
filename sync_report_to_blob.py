from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from cloud_sync import ReportOutputSync
from storage_backends import AzureBlobStorageBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bir rapora ait yerel ingestion çıktılarını "
            "Azure Blob Storage'a yükler."
        )
    )

    parser.add_argument(
        "--report-id",
        required=True,
        help="Yüklenecek raporun kimliği.",
    )
    parser.add_argument(
        "--data-directory",
        default="data",
        help="Yerel data klasörü. Varsayılan: data",
    )
    parser.add_argument(
        "--remote-prefix",
        default=None,
        help=(
            "Blob hedef prefix'i. Verilmezse "
            "reports/<report_id> kullanılır."
        ),
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Var olan blob dosyalarının üzerine yazmaz.",
    )

    return parser


def main() -> int:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    connection_string = os.getenv(
        "AZURE_STORAGE_CONNECTION_STRING",
        "",
    ).strip()
    container_name = os.getenv(
        "AZURE_STORAGE_CONTAINER",
        "",
    ).strip()

    if not connection_string:
        print(
            "Hata: AZURE_STORAGE_CONNECTION_STRING tanımlı değil.",
            file=sys.stderr,
        )
        return 2

    if not container_name:
        print(
            "Hata: AZURE_STORAGE_CONTAINER tanımlı değil.",
            file=sys.stderr,
        )
        return 2

    try:
        storage = AzureBlobStorageBackend(
            connection_string=connection_string,
            container_name=container_name,
        )

        sync_service = ReportOutputSync(
            storage_backend=storage,
            data_directory=Path(args.data_directory),
        )

        result = sync_service.sync_report(
            report_id=args.report_id,
            remote_prefix=args.remote_prefix,
            overwrite=not args.no_overwrite,
        )

    except Exception as exc:
        print(
            f"Blob senkronizasyonu başarısız: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())