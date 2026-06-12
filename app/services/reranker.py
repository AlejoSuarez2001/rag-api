import logging
import httpx
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker via HTTP remote service."""

    def __init__(self, reranker_url: str, top_k: int, timeout: int) -> None:
        self._url = reranker_url
        self._top_k = top_k
        self._timeout = timeout

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return chunks sorted by relevance score via remote HTTP service, limited to top_k."""
        if not chunks:
            return chunks

        pairs = [{"query": query, "text": chunk.text} for chunk in chunks]

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(self._url, json={"pairs": pairs})
                response.raise_for_status()
                scores: list[float] = response.json()["scores"]
        except Exception:
            logger.warning("Reranking failed, returning original order", exc_info=True)
            return chunks[: self._top_k]

        ranked = sorted(
            zip(scores, chunks),
            key=lambda t: t[0],
            reverse=True,
        )

        reranked = [
            chunk.model_copy(update={"score": float(score)})
            for score, chunk in ranked[: self._top_k]
        ]

        logger.debug(
            "Reranked %d → %d chunks. Top score: %.4f",
            len(chunks),
            len(reranked),
            reranked[0].score if reranked else 0,
        )
        return reranked
