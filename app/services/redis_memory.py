import json
import uuid
from datetime import datetime, timezone
import redis.asyncio as aioredis
from app.config import Settings
from app.models.schemas import Message, ConversationHistory, SharedConversation


class RedisMemory:
    def __init__(self, settings: Settings) -> None:
        self._client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True,
        )
        self._ttl = settings.redis_ttl_seconds
        self._share_ttl = settings.share_ttl_seconds
        self._max_messages = settings.max_history_messages

    async def get_history(self, conversation_id: str) -> ConversationHistory:
        raw = await self._client.get(self._key(conversation_id))
        if not raw:
            return ConversationHistory(conversation_id=conversation_id)
        data = json.loads(raw)
        return ConversationHistory(**data)

    async def add_turn(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        username: str | None = None,
        sources: list[str] | None = None,
    ) -> None:
        history = await self.get_history(conversation_id)

        if username and not history.username:
            history.username = username

        history.messages.append(Message(role="user", content=question))
        history.messages.append(Message(role="assistant", content=answer, sources=sources or []))

        # Keep only the last N messages to avoid token bloat
        history.messages = history.messages[-self._max_messages :]
        history.updated_at = datetime.now(timezone.utc).isoformat()

        await self._client.setex(
            self._key(conversation_id),
            self._ttl,
            history.model_dump_json(),
        )

        if username:
            user_key = self._user_key(username)
            await self._client.sadd(user_key, conversation_id)
            await self._client.expire(user_key, self._ttl)

    async def create_share(self, conversation_id: str) -> SharedConversation | None:
        history = await self.get_history(conversation_id)
        if not history.messages:
            return None
        token = uuid.uuid4().hex
        snapshot = SharedConversation(
            share_token=token,
            messages=history.messages,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await self._client.setex(
            self._share_key(token),
            self._share_ttl,
            snapshot.model_dump_json(),
        )
        return snapshot

    async def get_share(self, token: str) -> SharedConversation | None:
        raw = await self._client.get(self._share_key(token))
        if not raw:
            return None
        return SharedConversation(**json.loads(raw))

    async def get_user_conversations(self, username: str) -> list[str]:
        members = await self._client.smembers(self._user_key(username))
        return list(members)

    async def clear(self, conversation_id: str) -> None:
        await self._client.delete(self._key(conversation_id))

    async def health_check(self) -> bool:
        try:
            return await self._client.ping()
        except Exception:
            return False

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"rag:conv:{conversation_id}"

    @staticmethod
    def _user_key(username: str) -> str:
        return f"rag:user:{username}:convs"

    @staticmethod
    def _share_key(token: str) -> str:
        return f"rag:share:{token}"
