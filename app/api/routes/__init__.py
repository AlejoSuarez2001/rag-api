from fastapi import APIRouter
from app.api.routes import chat, conversations, health, share

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(conversations.router)
api_router.include_router(share.router)
