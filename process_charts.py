from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from pipelines.page_chart_processing_pipeline import (
    PageChartProcessingPipeline,
    PageChartProcessingRequest,
)
from storage.database import Database
from storage.vector_store import (
    HashingEmbeddingProvider,
    VectorStore,
    VectorStoreConfig,
)
from table_processing.vlm_table_fallback import VLMTableFallback


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="process-charts",
        description=(
            "Ingestion tarafından çıkarılmış tam sayfa grafik adaylarını "
            "PDF'yi yeniden işlemeden Groq Vision ile işler."
        ),
    )
    parser.add_argument(
        "--report-id",
        required=True,
        help="Veritabanındaki report_uid.",
    )
    parser.add_argument(
        "--data-directory",
        default="data",
        help="Uygulama veri klasörü.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help=(
            "Bu çalıştırmada işlenecek en fazla aday. "
            "0 bütün bekleyen adayları işler."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Durum dosyasını dikkate almadan adayları yeniden işler.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Daha önce failed olan adayları yeniden dener.",
    )
    parser.add_argument(
        "--retry-no-points",
        action="store_true",
        help="Daha önce no_points olan adayları yeniden dener.",
    )
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="Daha önce invalid olan adayları yeniden dener.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=4.0,
        help="Başarılı/başarısız Groq çağrıları arasındaki asgari bekleme.",
    )
    parser.add_argument(
        "--rate-limit-wait-seconds",
        type=float,
        default=65.0,
        help="Retry-After yoksa 429 sonrası beklenecek süre.",
    )
    parser.add_argument(
        "--stop-on-rate-limit",
        action="store_true",
        help="İlk 429 yanıtında worker'ı durdurur.",
    )
    return parser


def main() -> int:
    load_dotenv(dotenv_path=".env")

    args = create_parser().parse_args()
    data_directory = Path(
        args.data_directory
    ).expanduser().resolve()

    database = Database(
        database_path=(
            data_directory
            / "sustainability.db"
        )
    )

    vector_store = VectorStore(
        config=VectorStoreConfig(
            collection_name=(
                "sustainability_documents"
            ),
            persist_directory=(
                data_directory
                / "vectors"
            ),
        ),
        embedding_provider=(
            HashingEmbeddingProvider(
                dimension=384
            )
        ),
    )

    pipeline = PageChartProcessingPipeline(
        data_directory=data_directory,
        database=database,
        vector_store=vector_store,
        vlm_table_fallback=VLMTableFallback(),
    )

    try:
        result = pipeline.run(
            PageChartProcessingRequest(
                report_id=args.report_id,
                max_candidates=max(
                    args.max_candidates,
                    0,
                ),
                resume=not args.no_resume,
                retry_failed=args.retry_failed,
                retry_no_points=(
                    args.retry_no_points
                ),
                retry_invalid=args.retry_invalid,
                wait_between_calls_seconds=max(
                    args.wait_seconds,
                    0.0,
                ),
                fallback_rate_limit_wait_seconds=max(
                    args.rate_limit_wait_seconds,
                    1.0,
                ),
                stop_on_rate_limit=(
                    args.stop_on_rate_limit
                ),
            )
        )
        print(
            json.dumps(
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if not result.errors else 2
    finally:
        database.close()
        close_vector = getattr(
            vector_store,
            "close",
            None,
        )
        if callable(close_vector):
            close_vector()


if __name__ == "__main__":
    raise SystemExit(main())