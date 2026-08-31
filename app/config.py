"""
Central configuration. Everything tunable lives here and is overridable
via environment variables / .env — nothing is hardcoded in the pipeline
modules themselves, so swapping an embedding model or chunk size never
requires touching retrieval/generation code.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Embeddings
    embedding_provider: str = "local"          # local | openai
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_api_key: str = ""

    # Generation
    generation_provider: str = "anthropic"     # anthropic | openai | ollama
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    openai_generation_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"

    # Chunking
    chunk_strategy: str = "recursive"          # fixed | recursive | semantic
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_chunk_threshold: float = 0.55

    # Retrieval
    vector_top_k: int = 20
    keyword_top_k: int = 20
    hybrid_top_k: int = 10
    rrf_k: int = 60
    final_top_k: int = 4
    rerank_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Eval
    ragas_enabled: bool = False

    # Storage
    index_dir: str = "./storage"

    # Uploads
    max_upload_size_mb: int = 20
    upload_dir: str = "./uploads"   # originals are kept here after ingestion, not deleted

    # CORS — needed for a separately-hosted frontend (e.g. React dev server
    # on a different port) to call this API from the browser at all.
    # Comma-separated list of allowed origins, or "*" for all (dev only —
    # tighten this to your actual frontend origin(s) before deploying).
    cors_origins: str = "*"


settings = Settings()
