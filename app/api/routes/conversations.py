from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.models.schemas import ConversationHistory, ConversationListResponse, ConversationSummary
from app.security import get_current_token_payload
from app.services.redis_memory import RedisMemory

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


def get_redis_memory(settings: Settings = Depends(get_settings)) -> RedisMemory:
    return RedisMemory(settings)


@router.get(
    "",
    response_model=ConversationListResponse,
    summary="List all conversations for the authenticated user",
)
async def list_conversations(
    payload: dict[str, Any] = Depends(get_current_token_payload),
    memory: RedisMemory = Depends(get_redis_memory),
) -> ConversationListResponse:
    username: str | None = payload.get("preferred_username")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se pudo obtener el username del token JWT.",
        )
    conversation_ids = await memory.get_user_conversations(username)

    summaries = []
    for cid in conversation_ids:
        history = await memory.get_history(cid)
        first_assistant = next(
            (m.content for m in history.messages if m.role == "assistant"), None
        )
        preview = first_assistant[:150] if first_assistant else None
        search_text = " ".join(m.content for m in history.messages)
        summaries.append(ConversationSummary(
            conversation_id=cid,
            preview=preview,
            search_text=search_text,
            updated_at=history.updated_at,
        ))

    summaries.sort(key=lambda s: s.updated_at or "", reverse=True)
    return ConversationListResponse(username=username, conversations=summaries)


@router.get(
    "/{conversation_id}",
    response_model=ConversationHistory,
    summary="Get a conversation by its ID",
)
async def get_conversation(
    conversation_id: str,
    payload: dict[str, Any] = Depends(get_current_token_payload),
    memory: RedisMemory = Depends(get_redis_memory),
) -> ConversationHistory:
    username: str | None = payload.get("preferred_username")
    history = await memory.get_history(conversation_id)

    if not history.messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversación '{conversation_id}' no encontrada.",
        )

    if history.username and username and history.username != username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes acceso a esta conversación.",
        )

    return history
