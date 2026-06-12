import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f".env.{os.getenv('ENV', 'dev')}",
        env_file_encoding="utf-8"
    )
    # Auth / Keycloak
    auth_certs: str = "https://auth-dev.frba.utn.edu.ar/realms/frba/protocol/openid-connect/certs"
    auth_server_issuer: str = "https://auth-dev.frba.utn.edu.ar/realms/frba"
    auth_verify_ssl: bool = False
    keycloak_client_id: str = "utenia-llm"  # client del frontend

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_ttl_seconds: int = 3600
    share_ttl_seconds: int = 3600

    # Qdrant
    qdrant_host: str = "qdrant-rag"
    qdrant_port: int = 6333
    qdrant_collection: str = "tech_manuals"
    qdrant_top_k: int = 5

    # Ollama (generación de texto + embeddings)
    ollama_base_url: str
    ollama_model: str
    ollama_timeout: int = 120
    ollama_num_ctx: int = 8192

    # RAG Ingestion Service (interno)
    ingestion_service_url: str
    ingestion_timeout: int = 30

    # Embeddings (vía Ollama remoto)
    embedding_model: str
    embedding_dimensions: int

    # RAG
    max_context_chars: int = 12000
    max_context_tokens: int = 3000  # límite de tokens del contexto para el LLM
    max_history_messages: int = 6

    # Query rewriting
    query_rewrite_enabled: bool = True
    query_expansion_enabled: bool = True
    query_expansion_count: int = 3

    # Reranking (vía HTTP remoto)
    reranker_enabled: bool = True
    reranker_url: str
    reranker_timeout: int = 30
    reranker_top_k: int = 5       # final chunks after reranking
    retrieval_candidates: int = 15  # broader initial fetch before reranking
    retrieval_min_score: float = 0.01  # chunks below this score are discarded (RRF scores max ~0.033)

    # App
    app_name: str = "Hybrid RAG Support API"
    debug: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
