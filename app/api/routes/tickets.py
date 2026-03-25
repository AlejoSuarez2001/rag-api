from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.security import get_current_token_payload
from app.services.rag_service import RAGService

router = APIRouter(prefix="/tickets", tags=["Tickets"])


class TicketGenerateRequest(BaseModel):
    conversation_id: str


class TicketDescriptionResponse(BaseModel):
    description: str


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    return RAGService(settings)


@router.post("/generate-description", response_model=TicketDescriptionResponse)
async def generate_ticket_description(
    request: TicketGenerateRequest,
    payload: dict[str, Any] = Depends(get_current_token_payload),
    rag: RAGService = Depends(get_rag_service),
) -> TicketDescriptionResponse:
    try:
        description = await rag.generate_ticket_description(request.conversation_id)
        return TicketDescriptionResponse(description=description)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Timeout generando descripción.") from exc
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No se pudo conectar al LLM.") from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
