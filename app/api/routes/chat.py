from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.models.schemas import ChatRequest, ChatResponse
from app.security import get_current_token_payload
from app.services.feedback_repository import FeedbackRepository
from app.services.rag_service import RAGService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    return RAGService(settings)


def get_feedback_repository_optional(request: Request) -> FeedbackRepository | None:
    return getattr(request.app.state, "feedback_repo", None)


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a question and receive a grounded answer",
)
async def chat(
    request: ChatRequest,
    payload: dict[str, Any] = Depends(get_current_token_payload),
    rag: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    username: str | None = payload.get("preferred_username")
    try:
        return await rag.chat(
            conversation_id=request.conversation_id,
            question=request.question,
            username=username,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input inválido: {exc}",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Timeout esperando respuesta del servicio upstream (LLM/Qdrant).",
        ) from exc
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No se pudo conectar a un servicio upstream (LLM o Qdrant).",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error del servicio upstream: HTTP {exc.response.status_code}.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en el pipeline RAG: {exc}",
        ) from exc


@router.post(
    "/stream",
    summary="Send a question and receive a streaming answer (SSE)",
)
async def chat_stream(
    request: ChatRequest,
    payload: dict[str, Any] = Depends(get_current_token_payload),
    rag: RAGService = Depends(get_rag_service),
    feedback_repo: FeedbackRepository | None = Depends(get_feedback_repository_optional),
) -> StreamingResponse:
    username: str | None = payload.get("preferred_username")
    if request.regenerate and feedback_repo is not None:
        # El voto previo apuntaba a una respuesta que vamos a descartar: lo borramos.
        await feedback_repo.delete_for_exchange(request.conversation_id, request.question)
    return StreamingResponse(
        rag.chat_stream(
            conversation_id=request.conversation_id,
            question=request.question,
            username=username,
            regenerate=request.regenerate,
        ),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )
