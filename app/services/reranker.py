import logging
import threading
from typing import ClassVar
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker using sentence-transformers.

    The model is loaded lazily on first use and cached as a class-level
    singleton so it is shared across all instances (one per worker process).
    """

    _model: ClassVar = None
    _lock: ClassVar = threading.Lock()

    def __init__(self, model_name: str, device: str, top_k: int) -> None:
        self._model_name = model_name
        self._device = device
        self._top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return chunks sorted by cross-encoder relevance score, limited to top_k."""
        if not chunks:
            return chunks

        model = self._get_model()
        pairs = [(query, chunk.text) for chunk in chunks]

        try:
            scores: list[float] = model.predict(pairs, show_progress_bar=False).tolist()
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

    # ------------------------------------------------------------------
    # Lazy singleton model loader
    # ------------------------------------------------------------------

    def _get_model(self):
        if Reranker._model is None:
            with Reranker._lock:
                if Reranker._model is None:
                    logger.info(
                        "Loading cross-encoder model: %s on device %s",
                        self._model_name,
                        self._device,
                    )
                    from sentence_transformers import CrossEncoder  # lazy import

                    Reranker._model = CrossEncoder(self._model_name, device=self._device)
                    logger.info("Cross-encoder model loaded successfully")
        return Reranker._model
