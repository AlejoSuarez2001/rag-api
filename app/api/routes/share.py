from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.models.schemas import SharedConversation
from app.security import get_current_token_payload
from app.services.redis_memory import RedisMemory

router = APIRouter(tags=["Share"])


def get_redis_memory(settings: Settings = Depends(get_settings)) -> RedisMemory:
    return RedisMemory(settings)


@router.post(
    "/conversations/{conversation_id}/share",
    response_model=SharedConversation,
    summary="Create a shareable snapshot of a conversation",
)
async def create_share(
    conversation_id: str,
    payload: dict[str, Any] = Depends(get_current_token_payload),
    memory: RedisMemory = Depends(get_redis_memory),
) -> SharedConversation:
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

    snapshot = await memory.create_share(conversation_id)
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo crear el link de compartir.",
        )
    return snapshot


@router.get(
    "/share/{token}",
    response_model=SharedConversation,
    summary="Get a shared conversation by token (public)",
)
async def get_share(
    token: str,
    memory: RedisMemory = Depends(get_redis_memory),
) -> SharedConversation:
    snapshot = await memory.get_share(token)
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El link no existe o expiró.",
        )
    return snapshot
