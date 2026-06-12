from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import Settings, get_settings
from app.security import get_current_token_payload, require_ingestion_admin
from app.services.ingestion_client import IngestionClient

router = APIRouter(
    prefix="/ingestion",
    tags=["Ingestion"],
)


def get_ingestion_client(settings: Settings = Depends(get_settings)) -> IngestionClient:
    return IngestionClient(settings)


@router.post(
    "/full",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Lanzar ingestion completa de BookStack",
)
async def trigger_full_ingestion(
    force: bool = Query(False, description="Re-ingestar todas las páginas ignorando cambios"),
    payload: dict[str, Any] = Depends(require_ingestion_admin),
    client: IngestionClient = Depends(get_ingestion_client),
):
    """
    Inicia el pipeline completo: BookStack → clean → chunk → embed → Qdrant.
    Requiere rol 'ingestion-admin' en Keycloak.
    Retorna 202 inmediatamente. Consultá GET /ingestion/status para el progreso.
    """
    try:
        result = await client.trigger_full_ingestion(force=force)
        return result
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No se pudo conectar a rag_ingestion_service.",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Timeout esperando respuesta de rag_ingestion_service.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        try:
            error_data = exc.response.json()
            error_detail = error_data.get("detail", exc.response.text)
        except Exception:
            error_detail = exc.response.text
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=error_detail,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error disparando ingestion: {exc}",
        ) from exc


@router.get(
    "/status",
    summary="Estado del último job de ingestion",
)
async def get_ingestion_status(
    payload: dict[str, Any] = Depends(require_ingestion_admin),
    client: IngestionClient = Depends(get_ingestion_client),
):
    """Devuelve el estado actual (idle / running / completed / failed) y estadísticas."""
    try:
        status_info = await client.get_ingestion_status()
        return status_info
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No se pudo conectar a rag_ingestion_service.",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Timeout esperando respuesta de rag_ingestion_service.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        try:
            error_data = exc.response.json()
            error_detail = error_data.get("detail", exc.response.text)
        except Exception:
            error_detail = exc.response.text
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=error_detail,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo estado: {exc}",
        ) from exc
