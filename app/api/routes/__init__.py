from fastapi import APIRouter
from app.api.routes import chat, conversations, feedback, gaps, health, ingestion, share, tickets

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(conversations.router)
api_router.include_router(ingestion.router)
api_router.include_router(share.router)
api_router.include_router(tickets.router)
api_router.include_router(feedback.router)
api_router.include_router(gaps.router)
