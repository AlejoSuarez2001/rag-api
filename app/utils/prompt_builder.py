import logging

import tiktoken

from app.models.schemas import Message, RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un asistente técnico especializado en soporte. \
Tu tarea es responder preguntas basándote ÚNICAMENTE en los fragmentos de \
documentación técnica proporcionados en el contexto.

Reglas:
- Responde siempre en el mismo idioma de la pregunta.
- Si la información no está en el contexto, di claramente que no tienes esa información.
- Sé preciso, claro y conciso.
- Cita el manual o sección cuando sea relevante.
"""

_ENCODING_NAME = "cl100k_base"


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base, compatible with llama-based models)."""
    try:
        enc = tiktoken.get_encoding(_ENCODING_NAME)
        return len(enc.encode(text))
    except Exception:
        # Fallback: approximate as chars / 4
        return len(text) // 4


def build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[Message],
    max_context_chars: int = 12000,
    max_context_tokens: int = 3000,
) -> str:
    context = _build_context(chunks, max_context_chars, max_context_tokens)
    history_text = _build_history(history)

    parts = [SYSTEM_PROMPT]

    if history_text:
        parts.append(f"\n### Historial de conversación:\n{history_text}")

    parts.append(f"\n### Contexto de documentación técnica:\n{context}")
    parts.append(f"\n### Pregunta del usuario:\n{question}")
    parts.append("\n### Respuesta:")

    prompt = "\n".join(parts)

    total_tokens = count_tokens(prompt)
    logger.debug("Prompt total: %d tokens", total_tokens)

    return prompt


def _build_context(
    chunks: list[RetrievedChunk],
    max_chars: int,
    max_tokens: int,
) -> str:
    lines: list[str] = []
    total_chars = 0
    total_tokens = 0

    for i, chunk in enumerate(chunks, start=1):
        entry = f"[Fragmento {i} - {chunk.source}]\n{chunk.text}"
        entry_tokens = count_tokens(entry)

        if total_chars + len(entry) > max_chars:
            break
        if total_tokens + entry_tokens > max_tokens:
            logger.debug(
                "Contexto truncado en fragmento %d: %d tokens acumulados (límite %d)",
                i, total_tokens + entry_tokens, max_tokens,
            )
            break

        lines.append(entry)
        total_chars += len(entry)
        total_tokens += entry_tokens

    logger.debug("Contexto: %d fragmentos, %d tokens, %d chars", len(lines), total_tokens, total_chars)
    return "\n\n".join(lines) if lines else "No se encontró documentación relevante."


def _build_history(messages: list[Message]) -> str:
    if not messages:
        return ""
    return "\n".join(
        f"{'Usuario' if m.role == 'user' else 'Asistente'}: {m.content}"
        for m in messages
    )
