import asyncio
import logging
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchText,
)
from app.config import Settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class RetrievalService:
    """Hybrid retrieval: dense vector search + sparse keyword search via Qdrant."""

    def __init__(self, settings: Settings) -> None:
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._collection = settings.qdrant_collection
        self._top_k = settings.qdrant_top_k

    async def validate_collection_dimensions(self, expected_dim: int) -> None:
        """
        Validate that the Qdrant collection's vector size matches expected_dim.
        Should be called at application startup to catch embedding mismatches early.
        Raises RuntimeError if the dimensions don't match.
        """
        try:
            info = await self._client.get_collection(self._collection)
            vectors_config = info.config.params.vectors
            # vectors_config may be a dict (named vectors) or a VectorsConfig object
            if hasattr(vectors_config, "size"):
                actual_dim = vectors_config.size
            else:
                # Named vectors: use the first (and typically only) entry
                first = next(iter(vectors_config.values()))
                actual_dim = first.size

            if actual_dim != expected_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch en la colección '{self._collection}': "
                    f"Qdrant tiene vectores de {actual_dim} dims, "
                    f"pero el modelo produce {expected_dim} dims. "
                    "Re-ingestá todos los documentos con el modelo correcto."
                )
            logger.info(
                "Dimensiones validadas: colección '%s' tiene %d dims ✓",
                self._collection,
                actual_dim,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning(
                "No se pudo validar dimensiones de Qdrant (¿colección aún no creada?): %s", exc
            )

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        limit = top_k or self._top_k

        vector_results = await self._vector_search(embedding, limit)
        keyword_results = await self._keyword_search(query, limit)

        return self._merge_and_rank(vector_results, keyword_results, limit)

    async def multi_query_hybrid_search(
        self,
        queries: list[str],
        embeddings: list[list[float]],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Run hybrid search for each (query, embedding) pair in parallel,
        then fuse all result sets with RRF into a single ranked list.
        """
        limit = top_k or self._top_k
        # Fetch more candidates per query so fusion has enough material
        candidate_limit = limit * 2

        tasks = [
            self.hybrid_search(query=q, embedding=emb, top_k=candidate_limit)
            for q, emb in zip(queries, embeddings)
        ]
        per_query_results: list[list[RetrievedChunk]] = await asyncio.gather(*tasks)

        logger.debug(
            "Multi-query search: %d queries, candidate counts: %s",
            len(queries),
            [len(r) for r in per_query_results],
        )

        return self._multi_rrf(per_query_results, limit)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _vector_search(
        self, embedding: list[float], limit: int
    ) -> list[RetrievedChunk]:
        results = await self._client.search(
            collection_name=self._collection,
            query_vector=embedding,
            limit=limit,
            with_payload=True,
        )
        return [self._to_chunk(r) for r in results]

    async def _keyword_search(self, query: str, limit: int) -> list[RetrievedChunk]:
        """
        Keyword search using Qdrant's full-text filter on the 'text' payload field.
        Requires a text index on the collection (created during ingestion).
        """
        keyword_filter = Filter(
            must=[
                FieldCondition(
                    key="text",
                    match=MatchText(text=query),
                )
            ]
        )
        results = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=keyword_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        points, _ = results
        return [
            RetrievedChunk(
                text=p.payload.get("text", ""),
                source=p.payload.get("source", "unknown"),
                score=0.5,  # keyword matches get a base score
                chunk_id=str(p.id),
            )
            for p in points
        ]

    def _merge_and_rank(
        self,
        vector_chunks: list[RetrievedChunk],
        keyword_chunks: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion (RRF) to combine both result sets."""
        rrf_k = 60
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(vector_chunks):
            key = chunk.chunk_id or chunk.text[:64]
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            chunks_by_id[key] = chunk

        for rank, chunk in enumerate(keyword_chunks):
            key = chunk.chunk_id or chunk.text[:64]
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            chunks_by_id[key] = chunk

        ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

        merged: list[RetrievedChunk] = []
        for key in ranked_keys[:limit]:
            chunk = chunks_by_id[key]
            merged.append(chunk.model_copy(update={"score": scores[key]}))

        return merged

    @staticmethod
    def _to_chunk(result) -> RetrievedChunk:
        payload = result.payload or {}
        return RetrievedChunk(
            text=payload.get("text", ""),
            source=payload.get("source", "unknown"),
            score=result.score,
            chunk_id=str(result.id),
        )

    def _multi_rrf(
        self,
        result_sets: list[list[RetrievedChunk]],
        limit: int,
    ) -> list[RetrievedChunk]:
        """RRF across N result sets (multi-query fusion)."""
        rrf_k = 60
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, RetrievedChunk] = {}

        for result_list in result_sets:
            for rank, chunk in enumerate(result_list):
                key = chunk.chunk_id or chunk.text[:64]
                scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
                chunks_by_id[key] = chunk

        ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [
            chunks_by_id[k].model_copy(update={"score": scores[k]})
            for k in ranked_keys[:limit]
        ]

    async def health_check(self) -> bool:
        try:
            collections = await self._client.get_collections()
            names = [c.name for c in collections.collections]
            return self._collection in names
        except Exception:
            return False
