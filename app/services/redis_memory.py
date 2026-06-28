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
        no_info: bool = False,
    ) -> None:
        history = await self.get_history(conversation_id)

        if username and not history.username:
            history.username = username

        history.messages.append(Message(role="user", content=question))
        history.messages.append(Message(role="assistant", content=answer, sources=sources or [], no_info=no_info))

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

    async def pop_last_turn(self, conversation_id: str) -> str | None:
        """Quita el último turno (assistant + el user previo) y devuelve la pregunta de ese
        user, para poder regenerar la respuesta con el contexto previo intacto. Devuelve None
        si no hay nada que popear."""
        history = await self.get_history(conversation_id)
        msgs = history.messages
        if not msgs:
            return None
        if msgs[-1].role == "assistant":
            msgs.pop()
        question: str | None = None
        if msgs and msgs[-1].role == "user":
            question = msgs[-1].content
            msgs.pop()
        history.updated_at = datetime.now(timezone.utc).isoformat()
        # Mantenemos la key viva (preserva título) aunque quede sin mensajes: el add_turn de la
        # regeneración la vuelve a poblar inmediatamente.
        await self._client.setex(
            self._key(conversation_id),
            self._ttl,
            history.model_dump_json(),
        )
        return question

    async def set_title(self, conversation_id: str, title: str) -> None:
        history = await self.get_history(conversation_id)
        if not history.messages:
            return
        history.title = title
        await self._client.setex(
            self._key(conversation_id),
            self._ttl,
            history.model_dump_json(),
        )

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

    async def discard_conversation(self, username: str, conversation_id: str) -> None:
        """Quita un ID huérfano (cuya data ya expiró por TTL) del índice del usuario."""
        await self._client.srem(self._user_key(username), conversation_id)

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
