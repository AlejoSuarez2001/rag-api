import json
import logging
from collections.abc import AsyncIterator

import httpx
from app.config import Settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Handles text generation (via Ollama) and embedding (via Ollama remote).
    """

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_model
        self._embedding_model = settings.embedding_model
        self._timeout = settings.ollama_timeout
        self._num_ctx = settings.ollama_num_ctx

    async def generate(self, messages: list[dict], *, log_request: bool = False) -> str:
        """Send messages to Ollama /api/chat and return the generated text."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": self._num_ctx},
        }
        if log_request:
            logger.info(
                "Final Ollama chat payload:\n%s",
                json.dumps(payload, ensure_ascii=False, indent=2),
            )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

    async def generate_stream(self, messages: list[dict], *, log_request: bool = False) -> AsyncIterator[str]:
        """Stream tokens from Ollama /api/chat one by one."""
        payload = {"model": self._model, "messages": messages, "stream": True, "options": {"num_ctx": self._num_ctx}}
        if log_request:
            logger.info(
                "Final Ollama chat payload:\n%s",
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if token := data.get("message", {}).get("content"):
                            yield token
                        if data.get("done"):
                            break

    async def embed(self, text: str) -> list[float]:
        """Generate a dense embedding via Ollama remote API."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._embedding_model, "input": text},
            )
            response.raise_for_status()
            return response.json()["embeddings"][0]

    async def health_check(self) -> bool:
        """Check that Ollama is reachable for text generation."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
