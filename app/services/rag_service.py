import asyncio
import logging
from app.config import Settings
from app.models.schemas import ChatResponse, ConversationHistory, RetrievedChunk
from app.services.llm_service import LLMService
from app.services.query_rewriter import QueryRewriter
from app.services.reranker import Reranker
from app.services.redis_memory import RedisMemory
from app.services.retrieval_service import RetrievalService
from app.utils.prompt_builder import build_prompt

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrates the full RAG pipeline for a single chat turn."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = LLMService(settings)
        self._retrieval = RetrievalService(settings)
        self._memory = RedisMemory(settings)
        self._rewriter = QueryRewriter(self._llm, settings.query_expansion_count)
        self._reranker = (
            Reranker(settings.reranker_model, settings.reranker_top_k)
            if settings.reranker_enabled
            else None
        )

    async def chat(self, conversation_id: str, question: str) -> ChatResponse:
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
            logger.info("Reranked to %d chunks", len(chunks))
        else:
            chunks = candidates[: self._settings.reranker_top_k]

        # 6. Build prompt
        prompt = build_prompt(
            question=question,
            chunks=chunks,
            history=history.messages,
            max_context_chars=self._settings.max_context_chars,
            max_context_tokens=self._settings.max_context_tokens,
        )

        # 7. Call LLM
        answer = await self._llm.generate(prompt)

        # 8. Persist turn to Redis
        await self._memory.add_turn(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
        )

        # 9. Collect unique sources
        sources = list(dict.fromkeys(c.source for c in chunks))

        return ChatResponse(answer=answer, sources=sources)

    def _filter_chunks(
        self,
        chunks: list[RetrievedChunk],
        min_score: float = 0.001,
    ) -> list[RetrievedChunk]:
        return [c for c in chunks if c.score >= min_score and c.text.strip()]
