from functools import lru_cache
from fastapi import APIRouter, Depends, HTTPException, status
from app.config import Settings, get_settings
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["Chat"])


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    return RAGService(settings)


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a question and receive a grounded answer",
)
async def chat(
    request: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    try:
        return await rag.chat(
            conversation_id=request.conversation_id,
            question=request.question,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {exc}",
        ) from exc
