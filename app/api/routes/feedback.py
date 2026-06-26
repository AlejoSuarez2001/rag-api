from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.config import Settings, get_settings
from app.models.schemas import (
    FeedbackCreate,
    FeedbackCreatedResponse,
    FeedbackListResponse,
    FeedbackStats,
)
from app.security import get_current_token_payload, require_feedback_admin
from app.services.feedback_repository import FeedbackRepository

router = APIRouter(prefix="/feedback", tags=["Feedback"])


def get_feedback_repository(request: Request) -> FeedbackRepository:
    repo = getattr(request.app.state, "feedback_repo", None)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El servicio de feedback no está disponible (sin conexión a Postgres).",
        )
    return repo


@router.post(
    "",
    response_model=FeedbackCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Registrar feedback de usuario sobre una respuesta",
)
async def create_feedback(
    feedback: FeedbackCreate,
    payload: dict[str, Any] = Depends(get_current_token_payload),
    settings: Settings = Depends(get_settings),
    repo: FeedbackRepository = Depends(get_feedback_repository),
) -> FeedbackCreatedResponse:
    username: str | None = payload.get("preferred_username")
    try:
        new_id = await repo.insert(feedback, username=username, model=settings.ollama_model)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No se pudo guardar el feedback: {exc}",
        ) from exc
    return FeedbackCreatedResponse(id=new_id)


@router.get(
    "",
    response_model=FeedbackListResponse,
    summary="Listar feedback (admin)",
)
async def list_feedback(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    rating: Optional[int] = Query(None, ge=-1, le=1),
    category: list[str] = Query(default_factory=list),
    payload: dict[str, Any] = Depends(require_feedback_admin),
    repo: FeedbackRepository = Depends(get_feedback_repository),
) -> FeedbackListResponse:
    total, items = await repo.list(limit=limit, offset=offset, rating=rating, categories=category)
    return FeedbackListResponse(total=total, items=items)


@router.get(
    "/stats",
    response_model=FeedbackStats,
    summary="Estadísticas agregadas de feedback (admin)",
)
async def feedback_stats(
    payload: dict[str, Any] = Depends(require_feedback_admin),
    repo: FeedbackRepository = Depends(get_feedback_repository),
) -> FeedbackStats:
    return await repo.stats()
