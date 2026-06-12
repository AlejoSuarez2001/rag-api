import logging
from typing import Any

import httpx
from app.config import Settings

logger = logging.getLogger(__name__)


class IngestionClient:
    """Cliente para llamar a rag_ingestion_service internamente."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ingestion_service_url
        self._timeout = settings.ingestion_timeout

    async def trigger_full_ingestion(self, force: bool = False) -> dict[str, Any]:
        """Dispara la ingestion completa en rag_ingestion_service."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/v1/ingestion/full",
                params={"force": force},
            )
            response.raise_for_status()
            return response.json()

    async def get_ingestion_status(self) -> dict[str, Any]:
        """Obtiene el estado de la última ingestion."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{self._base_url}/api/v1/ingestion/status")
            response.raise_for_status()
            return response.json()
