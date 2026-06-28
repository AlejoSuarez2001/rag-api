from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.models.schemas import GapListResponse, GapStats
from app.security import require_feedback_admin
from app.services.gap_repository import GapRepository

router = APIRouter(prefix="/gaps", tags=["Gaps"])


def get_gap_repository(request: Request) -> GapRepository:
    repo = getattr(request.app.state, "gap_repo", None)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El servicio de gaps no está disponible (sin conexión a Postgres).",
        )
    return repo


@router.get(
    "",
    response_model=GapListResponse,
    summary="Listar gaps de documentación (admin)",
)
async def list_gaps(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    payload: dict[str, Any] = Depends(require_feedback_admin),
    repo: GapRepository = Depends(get_gap_repository),
) -> GapListResponse:
    total, items = await repo.list(limit=limit, offset=offset)
    return GapListResponse(total=total, items=items)


@router.get(
    "/stats",
    response_model=GapStats,
    summary="Estadísticas agregadas de gaps (admin)",
)
async def gap_stats(
    payload: dict[str, Any] = Depends(require_feedback_admin),
    repo: GapRepository = Depends(get_gap_repository),
) -> GapStats:
    return await repo.stats()
