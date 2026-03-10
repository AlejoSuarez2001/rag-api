from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_ttl_seconds: int = 3600

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "tech_manuals"
    qdrant_top_k: int = 5

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout: int = 120

    # RAG
    max_context_chars: int = 4000
    max_history_messages: int = 6

    # Query rewriting
    query_rewrite_enabled: bool = True
    query_expansion_enabled: bool = True
    query_expansion_count: int = 3

    # Reranking
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5       # final chunks after reranking
    retrieval_candidates: int = 15  # broader initial fetch before reranking

    # App
    app_name: str = "Hybrid RAG Support API"
    debug: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
