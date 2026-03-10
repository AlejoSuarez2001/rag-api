from app.models.schemas import Message, RetrievedChunk


SYSTEM_PROMPT = """Eres un asistente técnico especializado en soporte. \
Tu tarea es responder preguntas basándote ÚNICAMENTE en los fragmentos de \
documentación técnica proporcionados en el contexto.

Reglas:
- Responde siempre en el mismo idioma de la pregunta.
- Si la información no está en el contexto, di claramente que no tienes esa información.
- Sé preciso, claro y conciso.
- Cita el manual o sección cuando sea relevante.
"""


def build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[Message],
    max_context_chars: int = 4000,
) -> str:
    context = _build_context(chunks, max_context_chars)
    history_text = _build_history(history)

    parts = [SYSTEM_PROMPT]

    if history_text:
        parts.append(f"\n### Historial de conversación:\n{history_text}")

    parts.append(f"\n### Contexto de documentación técnica:\n{context}")
    parts.append(f"\n### Pregunta del usuario:\n{question}")
    parts.append("\n### Respuesta:")

    return "\n".join(parts)


def _build_context(chunks: list[RetrievedChunk], max_chars: int) -> str:
    lines: list[str] = []
    total = 0

    for i, chunk in enumerate(chunks, start=1):
        entry = f"[Fragmento {i} - {chunk.source}]\n{chunk.text}"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)

    return "\n\n".join(lines) if lines else "No se encontró documentación relevante."


def _build_history(messages: list[Message]) -> str:
    if not messages:
        return ""
    return "\n".join(
        f"{'Usuario' if m.role == 'user' else 'Asistente'}: {m.content}"
        for m in messages
    )
