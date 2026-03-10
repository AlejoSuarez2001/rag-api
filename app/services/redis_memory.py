import json
import redis.asyncio as aioredis
from app.config import Settings
from app.models.schemas import Message, ConversationHistory


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
    ) -> None:
        history = await self.get_history(conversation_id)

        history.messages.append(Message(role="user", content=question))
        history.messages.append(Message(role="assistant", content=answer))

        # Keep only the last N messages to avoid token bloat
        history.messages = history.messages[-self._max_messages :]

        await self._client.setex(
            self._key(conversation_id),
            self._ttl,
            history.model_dump_json(),
        )

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
