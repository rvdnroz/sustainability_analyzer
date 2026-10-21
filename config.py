from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent

LLMProviderName = Literal[
    "gemini",
    "groq",
    "openrouter",
    "ollama",
    "openai_compatible",
]

VisionProviderName = Literal[
    "groq",
    "openrouter",
    "openai_compatible",
]

JsonModePolicy = Literal[
    "auto",
    "required",
    "disabled",
]

_DEFAULT_LLM_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "groq": "qwen/qwen3.6-27b",
    "openrouter": "openrouter/free",
    "ollama": "qwen3.5:9b",
}

_DEFAULT_LLM_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}


class Settings(BaseSettings):
    """Sustainability Analyzer uygulamasının merkezi ayarları."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------
    # Genel uygulama ayarları
    # ---------------------------------------------------------

    app_name: str = "Sustainability Analyzer"

    app_environment: Literal[
        "development",
        "test",
        "production",
    ] = "development"

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    # ---------------------------------------------------------
    # API anahtarları
    # ---------------------------------------------------------

    exa_api_key: str | None = None

    # Provider bağımsız cevap modeli anahtarı. Tanımlanırsa sağlayıcıya özel
    # anahtarlardan önce kullanılır.
    llm_api_key: str | None = None

    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None

    # Görsel tablo fallback'i cevap modelinden bağımsız olarak hâlen Gemini
    # kullanır. Ayrı anahtar verilmezse GEMINI_API_KEY kullanılır.
    vlm_gemini_api_key: str | None = None

    # ---------------------------------------------------------
    # Cevap modeli sağlayıcısı
    # ---------------------------------------------------------

    llm_provider: LLMProviderName = "groq"
    llm_model: str | None = None
    llm_base_url: str | None = None

    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_maximum_output_tokens: int = Field(default=2200, ge=1)
    llm_top_p: float | None = Field(default=None, ge=0.0, le=1.0)

    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_connect_timeout_seconds: float = Field(default=15.0, gt=0)
    llm_maximum_retries: int = Field(default=3, ge=0)
    llm_initial_retry_delay_seconds: float = Field(default=1.0, gt=0)
    llm_maximum_retry_delay_seconds: float = Field(default=20.0, gt=0)
    llm_retry_jitter_ratio: float = Field(default=0.20, ge=0.0, le=1.0)

    llm_json_mode_policy: JsonModePolicy = "auto"
    llm_verify_ssl: bool = True
    llm_trust_environment: bool = True

    openrouter_http_referer: str | None = None
    openrouter_app_title: str | None = "Sustainability Analyzer"

    # Geriye dönük Gemini ayarları. Gemini cevap sağlayıcısı seçildiğinde
    # llm_model verilmemişse gemini_model kullanılır.
    gemini_model: str = "gemini-2.5-flash"
    gemini_top_k: int | None = Field(default=None, ge=1)
    gemini_thinking_level: str | None = None

    # Yerel Ollama için varsayılan OpenAI-compatible adres.
    ollama_base_url: str = "http://localhost:11434/v1"

    # ---------------------------------------------------------
    # Multimodal vision cevap ayarları
    # ---------------------------------------------------------

    vision_fallback_enabled: bool = False
    vision_provider: VisionProviderName = "openrouter"
    vision_model: str | None = None
    vision_api_key: str | None = None
    vision_base_url: str | None = None

    vision_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    vision_maximum_output_tokens: int = Field(default=3200, ge=1)
    vision_maximum_images: int = Field(default=4, ge=1, le=16)
    vision_image_detail: Literal["auto", "low", "high"] = "high"
    vision_minimum_metric_confidence: float = Field(
        default=0.45, ge=0.0, le=1.0
    )
    vision_top_k_per_question: int = Field(default=3, ge=1)
    vision_cache_enabled: bool = True
    vision_require_pending_only: bool = True

    # ---------------------------------------------------------
    # Görsel tablo fallback ayarları
    # ---------------------------------------------------------

    vlm_fallback_enabled: bool = False
    vlm_gemini_model: str | None = None

    # ---------------------------------------------------------
    # Ana dosya dizinleri
    # ---------------------------------------------------------

    data_directory: Path = BASE_DIR / "data"
    database_path: Path = BASE_DIR / "data" / "sustainability_analyzer.db"

    # ---------------------------------------------------------
    # Exa ve rapor arama ayarları
    # ---------------------------------------------------------

    exa_result_limit: int = Field(default=10, ge=1, le=100)
    minimum_candidate_score: float = Field(default=0.45, ge=0.0, le=1.0)

    # ---------------------------------------------------------
    # HTTP ve PDF ayarları
    # ---------------------------------------------------------

    http_timeout_seconds: float = Field(default=45.0, gt=0)
    maximum_pdf_size_mb: int = Field(default=150, ge=1)
    minimum_pdf_page_count: int = Field(default=5, ge=1)

    # ---------------------------------------------------------
    # Sayfa işleme ayarları
    # ---------------------------------------------------------

    page_render_dpi: int = Field(default=150, ge=72, le=600)
    ocr_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    table_detection_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    # ---------------------------------------------------------
    # Semantic chunking ayarları
    # ---------------------------------------------------------

    chunk_size: int = Field(default=700, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)

    # ---------------------------------------------------------
    # Retrieval ayarları
    # ---------------------------------------------------------

    table_retrieval_top_k: int = Field(default=10, ge=1)
    bm25_retrieval_top_k: int = Field(default=20, ge=1)
    vector_retrieval_top_k: int = Field(default=20, ge=1)
    reranker_top_k: int = Field(default=8, ge=1)
    minimum_evidence_score: float = Field(default=0.45, ge=0.0, le=1.0)

    # ---------------------------------------------------------
    # Embedding ve reranker ayarları
    # ---------------------------------------------------------

    embedding_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    reranker_model_name: str = (
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )

    # ---------------------------------------------------------
    # Doğrulamalar
    # ---------------------------------------------------------

    @field_validator(
        "exa_api_key",
        "llm_api_key",
        "groq_api_key",
        "openrouter_api_key",
        "openai_api_key",
        "gemini_api_key",
        "vlm_gemini_api_key",
        "vision_api_key",
        "vision_model",
        "vision_base_url",
        "llm_model",
        "llm_base_url",
        "openrouter_http_referer",
        "openrouter_app_title",
        "gemini_thinking_level",
        "vlm_gemini_model",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("vision_provider", mode="before")
    @classmethod
    def normalize_vision_provider(cls, value: str) -> str:
        normalized = str(value).strip().casefold().replace("-", "_")
        aliases = {
            "open_router": "openrouter",
            "openai": "openai_compatible",
        }
        return aliases.get(normalized, normalized)

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_llm_provider(cls, value: str) -> str:
        normalized = str(value).strip().casefold().replace("-", "_")
        aliases = {
            "google": "gemini",
            "google_gemini": "gemini",
            "open_router": "openrouter",
            "local": "ollama",
            "openai": "openai_compatible",
        }
        return aliases.get(normalized, normalized)

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_overlap negatif olamaz.")
        return value

    # ---------------------------------------------------------
    # LLM çözümleme yardımcıları
    # ---------------------------------------------------------

    @property
    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model

        if self.llm_provider == "gemini" and self.gemini_model:
            return self.gemini_model

        default_model = _DEFAULT_LLM_MODELS.get(self.llm_provider)
        if default_model:
            return default_model

        raise RuntimeError(
            "LLM_MODEL tanımlanmamış. openai_compatible sağlayıcısı için "
            "model adı açıkça verilmelidir."
        )

    @property
    def resolved_llm_api_key(self) -> str | None:
        if self.llm_api_key:
            return self.llm_api_key

        provider_keys = {
            "gemini": self.gemini_api_key,
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
            "openai_compatible": self.openai_api_key,
            "ollama": None,
        }
        return provider_keys.get(self.llm_provider)

    @property
    def resolved_llm_base_url(self) -> str | None:
        if self.llm_base_url:
            return self.llm_base_url.rstrip("/")

        if self.llm_provider == "ollama":
            return self.ollama_base_url.rstrip("/")

        default_url = _DEFAULT_LLM_BASE_URLS.get(self.llm_provider)
        return default_url.rstrip("/") if default_url else None

    @property
    def resolved_vision_model(self) -> str:
        if self.vision_model:
            return self.vision_model
        raise RuntimeError(
            "VISION_MODEL tanımlanmamış. Vision fallback kullanılacaksa "
            "multimodal model adı açıkça verilmelidir."
        )

    @property
    def resolved_vision_api_key(self) -> str | None:
        if self.vision_api_key:
            return self.vision_api_key
        provider_keys = {
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
            "openai_compatible": self.openai_api_key,
        }
        return provider_keys.get(self.vision_provider)

    @property
    def resolved_vision_base_url(self) -> str | None:
        if self.vision_base_url:
            return self.vision_base_url.rstrip("/")
        default_url = _DEFAULT_LLM_BASE_URLS.get(self.vision_provider)
        return default_url.rstrip("/") if default_url else None

    @property
    def resolved_vlm_gemini_api_key(self) -> str | None:
        return self.vlm_gemini_api_key or self.gemini_api_key

    def require_llm_api_key(self) -> str:
        api_key = self.resolved_llm_api_key
        if api_key:
            return api_key

        environment_names = {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai_compatible": "LLM_API_KEY veya OPENAI_API_KEY",
        }
        variable_name = environment_names.get(self.llm_provider)

        if self.llm_provider == "ollama":
            return ""

        raise RuntimeError(
            f"{self.llm_provider} API anahtarı bulunamadı. "
            f"{variable_name or 'uygun API anahtarı'} tanımlanmalıdır."
        )

    # ---------------------------------------------------------
    # Alt dizin özellikleri
    # ---------------------------------------------------------

    @property
    def downloads_directory(self) -> Path:
        return self.data_directory / "downloads"

    @property
    def rendered_pages_directory(self) -> Path:
        return self.data_directory / "rendered_pages"

    @property
    def extracted_tables_directory(self) -> Path:
        return self.data_directory / "tables"

    @property
    def indexes_directory(self) -> Path:
        return self.data_directory / "indexes"

    @property
    def cache_directory(self) -> Path:
        return self.data_directory / "cache"

    @property
    def output_directory(self) -> Path:
        return self.data_directory / "output"

    @property
    def vectors_directory(self) -> Path:
        return self.data_directory / "vectors"

    @property
    def table_images_directory(self) -> Path:
        return self.data_directory / "table_images"

    @property
    def file_store_directory(self) -> Path:
        return self.data_directory / "files"

    # ---------------------------------------------------------
    # Yardımcı metotlar
    # ---------------------------------------------------------

    def create_runtime_directories(self) -> None:
        directories = [
            self.data_directory,
            self.downloads_directory,
            self.rendered_pages_directory,
            self.extracted_tables_directory,
            self.indexes_directory,
            self.cache_directory,
            self.output_directory,
            self.vectors_directory,
            self.table_images_directory,
            self.file_store_directory,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def require_exa_api_key(self) -> str:
        if self.exa_api_key is None:
            raise RuntimeError(
                "EXA_API_KEY tanımlanmamış. Proje dizinindeki .env "
                "dosyasına Exa API anahtarını ekleyin."
            )
        return self.exa_api_key

    def require_gemini_api_key(self) -> str:
        if self.gemini_api_key is None:
            raise RuntimeError(
                "GEMINI_API_KEY tanımlanmamış. Proje dizinindeki .env "
                "dosyasına Gemini API anahtarını ekleyin."
            )
        return self.gemini_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()

    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError(
            "chunk_overlap değeri chunk_size değerinden küçük olmalıdır."
        )

    if (
        settings.llm_provider == "openai_compatible"
        and settings.resolved_llm_base_url is None
    ):
        raise ValueError(
            "openai_compatible sağlayıcısı için LLM_BASE_URL tanımlanmalıdır."
        )

    settings.create_runtime_directories()
    return settings


if __name__ == "__main__":
    settings = get_settings()

    print("Yapılandırma başarıyla yüklendi.")
    print(f"Uygulama: {settings.app_name}")
    print(f"Ortam: {settings.app_environment}")
    print(f"Veri dizini: {settings.data_directory.resolve()}")
    print(f"LLM sağlayıcısı: {settings.llm_provider}")
    print(f"LLM modeli: {settings.resolved_llm_model}")
    print(f"LLM base URL: {settings.resolved_llm_base_url or '-'}")
    print(
        "VLM fallback: "
        f"{'açık' if settings.vlm_fallback_enabled else 'kapalı'}"
    )