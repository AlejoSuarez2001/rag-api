import asyncio
import json
import logging
import threading
from typing import ClassVar

import httpx
from app.config import Settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Handles text generation (via Ollama) and embedding (via sentence-transformers).
    The embedding model is a process-level singleton loaded lazily on first use.
    """

    _embed_model: ClassVar = None
    _embed_lock: ClassVar = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_model
        self._embed_model_name = settings.embedding_model
        self._embed_device = settings.embedding_device
        self._timeout = settings.ollama_timeout

    async def generate(self, prompt: str, *, log_request: bool = False) -> str:
        """Send a prompt to Ollama and return the generated text."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if log_request:
            logger.info(
                "Final Ollama generate payload:\n%s",
                json.dumps(payload, ensure_ascii=False, indent=2),
            )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            return response.json()["response"]

    async def embed(self, text: str) -> list[float]:
        """
        Generate a dense embedding using sentence-transformers.
        Runs the (synchronous) model in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode_sync, text)

    def _encode_sync(self, text: str) -> list[float]:
        model = self._get_embed_model()
        return model.encode(text, convert_to_numpy=True).tolist()

    def _get_embed_model(self):
        """Lazy singleton: load the sentence-transformer model once per process."""
        if LLMService._embed_model is None:
            with LLMService._embed_lock:
                if LLMService._embed_model is None:
                    logger.info("Loading embedding model: %s", self._embed_model_name)
                    from sentence_transformers import SentenceTransformer  # lazy import
                    LLMService._embed_model = SentenceTransformer(
                        self._embed_model_name, device=self._embed_device
                    )
                    logger.info("Embedding model loaded successfully")
        return LLMService._embed_model

    async def health_check(self) -> bool:
        """Check that Ollama is reachable for text generation."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
