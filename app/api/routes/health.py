from fastapi import APIRouter, Depends
from app.config import Settings, get_settings
from app.security import require_health_role
from app.services.llm_service import LLMService
from app.services.redis_memory import RedisMemory
from app.services.retrieval_service import RetrievalService

router = APIRouter(
    prefix="/health",
    tags=["Health"],
    dependencies=[Depends(require_health_role)],
)


@router.get("", summary="Liveness check")
async def health():
    return {"status": "ok"}


@router.get("/ready", summary="Readiness check — verifies all dependencies")
async def readiness(settings: Settings = Depends(get_settings)):
    results = {
        "redis": await RedisMemory(settings).health_check(),
        "qdrant": await RetrievalService(settings).health_check(),
        "ollama": await LLMService(settings).health_check(),
    }
    all_ok = all(results.values())
    return {"status": "ready" if all_ok else "degraded", "checks": results}
