import asyncio
import json
import logging
from app.models.schemas import Message
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

_STANDALONE_PROMPT = """Tu única tarea es resolver referencias implícitas al historial de conversación \
(pronombres como "eso", "lo mismo", "el anterior", "esa opción", etc.) en la pregunta del usuario.

REGLAS ESTRICTAS:
- Si la pregunta NO tiene referencias implícitas al historial, devuélvela EXACTAMENTE igual, sin cambiar ninguna palabra.
- Si tiene referencias, reemplaza SOLO esas referencias con el término concreto al que apuntan en el historial.
- NO agregues palabras, contexto, institución, ni información que no esté en la pregunta original.
- NO mejores, reformules ni expandas la pregunta.
- NO cambies el verbo ni la intención de la pregunta.

Historial:
{history}

Pregunta original: {question}

Responde ÚNICAMENTE con la pregunta resultante, sin explicaciones ni comillas."""

_EXPANSION_PROMPT = """Eres un motor de búsqueda técnica. Genera {n} variantes de búsqueda \
distintas para encontrar información relevante sobre la siguiente consulta en una base de \
conocimiento de manuales técnicos. Las variantes deben cubrir distintos ángulos y vocabulario.

Consulta: {query}

Responde ÚNICAMENTE con un JSON array de strings. Ejemplo: ["variante 1", "variante 2"]"""


class QueryRewriter:
    def __init__(self, llm: LLMService, expansion_count: int = 3) -> None:
        self._llm = llm
        self._expansion_count = expansion_count

    async def rewrite_standalone(
        self,
        question: str,
        history: list[Message],
    ) -> str:
        """Rewrite the question removing any implicit references to conversation history."""
        if not history:
            return question

        history_text = "\n".join(
            f"{'Usuario' if m.role == 'user' else 'Asistente'}: {m.content}"
            for m in history[-4:]  # last 2 turns is enough for context
        )
        prompt = _STANDALONE_PROMPT.format(history=history_text, question=question)

        try:
            rewritten = await self._llm.generate(prompt)
            rewritten = rewritten.strip().strip('"').strip("'")
            logger.info(
                "Ollama rewrote user query before embedding: original=%r rewritten=%r",
                question,
                rewritten,
            )
            return rewritten or question
        except Exception:
            logger.warning("Query rewrite failed, using original query", exc_info=True)
            return question

    async def expand_queries(self, query: str) -> list[str]:
        """Generate N search query variants to improve recall."""
        prompt = _EXPANSION_PROMPT.format(n=self._expansion_count, query=query)

        try:
            raw = await self._llm.generate(prompt)
            # Extract the JSON array from the response (LLMs sometimes add extra text)
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return [query]
            variants: list[str] = json.loads(raw[start:end])
            # Always include the original as first candidate
            all_queries = [query] + [v for v in variants if v and v != query]
            logger.debug("Expanded to %d queries", len(all_queries))
            return all_queries[: self._expansion_count + 1]
        except Exception:
            logger.warning("Query expansion failed, using single query", exc_info=True)
            return [query]
