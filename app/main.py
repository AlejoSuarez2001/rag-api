import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.api.routes import api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    from app.services.retrieval_service import RetrievalService
    retrieval = RetrievalService(settings)
    await retrieval.validate_collection_dimensions(settings.embedding_dimensions)
    yield


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Hybrid RAG API for technical support manuals",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
