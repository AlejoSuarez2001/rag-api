import asyncio
import json
import logging
from collections.abc import AsyncIterator

import httpx
from pydantic import ValidationError
from app.config import Settings
from app.models.schemas import ChatResponse, ConversationHistory, RetrievedChunk
from app.services.llm_service import LLMService
from app.services.query_rewriter import QueryRewriter
from app.services.reranker import Reranker
from app.services.redis_memory import RedisMemory
from app.services.retrieval_service import RetrievalService
from app.services.prompt_builder import build_prompt

logger = logging.getLogger(__name__)

_NO_INFO_MARKER = "[SIN_INFO]"

_NO_INFO_PHRASES = (
    "no tengo información",
    "no cuento con información",
    "no cuento con esa información",
    "no dispongo de información",
    "no encuentro información",
    "no tengo datos",
    "no tengo suficiente información",
    "información suficiente para responder",
    "no puedo responder",
    "no está en mi información",
    "no tengo info",
)


def _strip_no_info_marker(text: str) -> tuple[str, bool]:
    """Returns (clean_text, no_info). Marker takes priority; keywords as fallback."""
    stripped = text.lstrip()
    if stripped.startswith(_NO_INFO_MARKER):
        return stripped[len(_NO_INFO_MARKER):].lstrip(), True
    lower = text.lower()
    no_info = any(phrase in lower for phrase in _NO_INFO_PHRASES)
    return text, no_info


class RAGService:
    """Orchestrates the full RAG pipeline for a single chat turn."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = LLMService(settings)
        self._retrieval = RetrievalService(settings)
        self._memory = RedisMemory(settings)
        self._rewriter = QueryRewriter(self._llm, settings.query_expansion_count)
        self._reranker = (
            Reranker(
                settings.reranker_model,
                settings.reranker_device,
                settings.reranker_top_k,
            )
            if settings.reranker_enabled
            else None
        )

    async def chat(self, conversation_id: str, question: str, username: str | None = None) -> ChatResponse:
        # 1. Retrieve conversation history
        history = await self._memory.get_history(conversation_id)

        # 2. Query rewriting — remove implicit references to conversation history
        standalone_query = question
        if self._settings.query_rewrite_enabled:
            standalone_query = await self._rewriter.rewrite_standalone(
                question=question,
                history=history.messages,
            )
            logger.info("Standalone query: %r", standalone_query)

        # 3. Multi-query expansion + parallel embedding
        search_queries = [standalone_query]
        if self._settings.query_expansion_enabled:
            search_queries = await self._rewriter.expand_queries(standalone_query)
            logger.info("Expanded to %d queries", len(search_queries))

        logger.info("Queries sent to embedding model: %s", search_queries)
        embeddings = await asyncio.gather(
            *[self._llm.embed(q) for q in search_queries]
        )

        # 4. Multi-query hybrid search (fetches more candidates for reranking)
        candidates = await self._retrieval.multi_query_hybrid_search(
            queries=search_queries,
            embeddings=list(embeddings),
            top_k=self._settings.retrieval_candidates,
        )
        candidates = self._filter_chunks(candidates)
        logger.info("Retrieved %d candidates after filtering", len(candidates))

        # 5. Rerank candidates with cross-encoder
        if self._reranker and candidates:
            chunks = self._reranker.rerank(standalone_query, candidates)
            logger.info(
                "Reranked chunks (%d): %s",
                len(chunks),
                [
                    {"rank": i + 1, "chunk_id": c.chunk_id, "title": c.title, "source": c.source, "score": round(c.score, 4)}
                    for i, c in enumerate(chunks)
                ],
            )
        else:
            chunks = candidates[: self._settings.reranker_top_k]
            logger.info(
                "Reranker disabled — top chunks by retrieval score (%d): %s",
                len(chunks),
                [
                    {"rank": i + 1, "chunk_id": c.chunk_id, "title": c.title, "source": c.source, "score": round(c.score, 4)}
                    for i, c in enumerate(chunks)
                ],
            )
        chunks = await self._expand_section_chunks(
            selected_chunks=chunks,
            max_chunks=self._settings.reranker_top_k,
        )
        logger.info(
            "Final chunks sent to prompt after section expansion (%d): %s",
            len(chunks),
            [
                {"rank": i + 1, "chunk_id": c.chunk_id, "title": c.title, "source": c.source, "score": round(c.score, 4)}
                for i, c in enumerate(chunks)
            ],
        )

        # 6. Build prompt
        prompt = build_prompt(
            question=question,
            chunks=chunks,
            history=history.messages,
            max_context_chars=self._settings.max_context_chars,
            max_context_tokens=self._settings.max_context_tokens,
        )
        logger.info("Final answer prompt built (%d chars)", len(prompt))

        # 7. Call LLM
        raw_answer = await self._llm.generate(prompt, log_request=True)
        answer, no_info = _strip_no_info_marker(raw_answer)

        # 8. Collect unique sources
        sources = list(dict.fromkeys(c.source for c in chunks))

        # 9. Persist turn to Redis
        await self._memory.add_turn(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            username=username,
            sources=sources,
            no_info=no_info,
        )

        return ChatResponse(answer=answer, sources=sources, no_info=no_info)

    async def chat_stream(
        self, conversation_id: str, question: str, username: str | None = None
    ) -> AsyncIterator[str]:
        """Same pipeline as chat() but streams the LLM response token by token via SSE."""
        try:
            history = await self._memory.get_history(conversation_id)

            standalone_query = question
            if self._settings.query_rewrite_enabled:
                standalone_query = await self._rewriter.rewrite_standalone(
                    question=question, history=history.messages
                )

            search_queries = [standalone_query]
            if self._settings.query_expansion_enabled:
                search_queries = await self._rewriter.expand_queries(standalone_query)

            embeddings = await asyncio.gather(*[self._llm.embed(q) for q in search_queries])

            candidates = await self._retrieval.multi_query_hybrid_search(
                queries=search_queries,
                embeddings=list(embeddings),
                top_k=self._settings.retrieval_candidates,
            )
            candidates = self._filter_chunks(candidates)

            if self._reranker and candidates:
                chunks = self._reranker.rerank(standalone_query, candidates)
                logger.info(
                    "Reranked chunks (%d): %s",
                    len(chunks),
                    [
                        {"rank": i + 1, "chunk_id": c.chunk_id, "title": c.title, "source": c.source, "score": round(c.score, 4)}
                        for i, c in enumerate(chunks)
                    ],
                )
            else:
                chunks = candidates[: self._settings.reranker_top_k]
                logger.info(
                    "Reranker disabled — top chunks by retrieval score (%d): %s",
                    len(chunks),
                    [
                        {"rank": i + 1, "chunk_id": c.chunk_id, "title": c.title, "source": c.source, "score": round(c.score, 4)}
                        for i, c in enumerate(chunks)
                    ],
                )
            chunks = await self._expand_section_chunks(
                selected_chunks=chunks, max_chunks=self._settings.reranker_top_k
            )
            logger.info(
                "Final chunks sent to prompt after section expansion (%d): %s",
                len(chunks),
                [
                    {"rank": i + 1, "chunk_id": c.chunk_id, "title": c.title, "source": c.source, "score": round(c.score, 4)}
                    for i, c in enumerate(chunks)
                ],
            )

            prompt = build_prompt(
                question=question,
                chunks=chunks,
                history=history.messages,
                max_context_chars=self._settings.max_context_chars,
                max_context_tokens=self._settings.max_context_tokens,
            )

            full_answer = ""
            buffer = ""
            marker_checked = False

            async for token in self._llm.generate_stream(prompt, log_request=True):
                full_answer += token
                if not marker_checked:
                    buffer += token
                    if len(buffer) >= len(_NO_INFO_MARKER):
                        marker_checked = True
                        buffer, _ = _strip_no_info_marker(buffer)
                        if buffer:
                            yield f"data: {json.dumps({'token': buffer})}\n\n"
                else:
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # Flush buffer if stream ended before we had enough chars to check
            if not marker_checked and buffer:
                buffer, _ = _strip_no_info_marker(buffer)
                if buffer:
                    yield f"data: {json.dumps({'token': buffer})}\n\n"

            clean_answer, no_info = _strip_no_info_marker(full_answer)
            sources = list(dict.fromkeys(c.source for c in chunks))
            await self._memory.add_turn(
                conversation_id=conversation_id,
                question=question,
                answer=clean_answer,
                username=username,
                sources=sources,
                no_info=no_info,
            )

            yield f"data: {json.dumps({'done': True, 'sources': sources, 'no_info': no_info})}\n\n"

        except ValidationError as exc:
            logger.error("Validation error in chat_stream: %s", exc)
            yield f"data: {json.dumps({'error': f'Input inválido: {exc}'})}\n\n"
        except httpx.TimeoutException as exc:
            logger.error("Timeout in chat_stream: %s", exc)
            yield f"data: {json.dumps({'error': 'Timeout esperando respuesta del servicio upstream (LLM/Qdrant).'})}\n\n"
        except httpx.ConnectError as exc:
            logger.error("Connection error in chat_stream: %s", exc)
            yield f"data: {json.dumps({'error': 'No se pudo conectar a un servicio upstream (LLM o Qdrant).'})}\n\n"
        except httpx.HTTPStatusError as exc:
            logger.error("HTTP status error in chat_stream: %s", exc)
            yield f"data: {json.dumps({'error': f'Error del servicio upstream: HTTP {exc.response.status_code}.'})}\n\n"
        except Exception as exc:
            logger.error("Unexpected error in chat_stream: %s", exc)
            yield f"data: {json.dumps({'error': f'Error en el pipeline RAG: {exc}'})}\n\n"

    async def generate_ticket_description(self, conversation_id: str) -> str:
        history = await self._memory.get_history(conversation_id)
        messages = history.messages[-20:]

        conversation_text = "\n".join(
            f"{'Usuario' if m.role == 'user' else 'Asistente'}: {m.content}"
            for m in messages
        )

        prompt = (
            "Eres un asistente de soporte técnico.\n\n"
            "Analiza la conversación y genera un ticket.\n\n"
            "Reglas:\n"
            "- No inventes información.\n"
            "- Sé claro y conciso.\n"
            "- Usa solo la información disponible.\n"
            "- Escribe en español.\n"
            "- No uses markdown, asteriscos, negritas ni ningún tipo de formato especial. Solo texto plano.\n\n"
            "Formato obligatorio:\n\n"
            "Problema del usuario:\n"
            "- Qué quiere hacer y qué falla.\n\n"
            "Intento de resolución:\n"
            "- Qué se probó (si no hay, escribir \"No se especifica\").\n\n"
            "Tipo de problema:\n"
            "- Bug | Configuración | Acceso | VPN | Otro\n\n"
            "---\n\n"
            f"Conversación:\n{conversation_text}"
        )

        return await self._llm.generate(prompt)

    def _filter_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return [c for c in chunks if c.score >= self._settings.retrieval_min_score and c.text.strip()]

    async def _expand_section_chunks(
        self,
        selected_chunks: list[RetrievedChunk],
        max_chunks: int,
    ) -> list[RetrievedChunk]:
        if max_chunks <= 1:
            return selected_chunks[:max_chunks]

        result: list[RetrievedChunk] = []
        used_chunk_ids: set[str] = set()

        for chunk in selected_chunks:
            if len(result) >= max_chunks:
                break

            chunk_key = chunk.chunk_id
            if chunk_key and chunk_key not in used_chunk_ids:
                result.append(chunk)
                used_chunk_ids.add(chunk_key)

            if len(result) >= max_chunks or not chunk.source or not chunk.title:
                continue

            section_chunks = await self._retrieval.fetch_section_chunks(
                source=chunk.source,
                title=chunk.title,
            )
            companions = self._order_section_companions(chunk, section_chunks, used_chunk_ids)
            for companion in companions:
                if len(result) >= max_chunks:
                    break
                result.append(companion)
                if companion.chunk_id:
                    used_chunk_ids.add(companion.chunk_id)

        return result

    @staticmethod
    def _order_section_companions(
        chunk: RetrievedChunk,
        section_chunks: list[RetrievedChunk],
        used_chunk_ids: set[str],
    ) -> list[RetrievedChunk]:
        companions = [
            section_chunk
            for section_chunk in section_chunks
            if section_chunk.chunk_id
            and section_chunk.chunk_id not in used_chunk_ids
            and section_chunk.chunk_id != chunk.chunk_id
        ]
        if not companions:
            return []

        def companion_sort_key(candidate: RetrievedChunk) -> tuple[int, int, float]:
            if chunk.position is not None and candidate.position is not None:
                return (0, abs(candidate.position - chunk.position), -candidate.score)
            return (1, 0, -candidate.score)

        companions.sort(key=companion_sort_key)
        return companions
