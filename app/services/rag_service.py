import asyncio
import json
import logging
from collections.abc import AsyncIterator

import httpx
from pydantic import ValidationError
from app.config import Settings
from app.models.schemas import ChatResponse, ConversationHistory, RetrievedChunk
from app.services.gap_repository import GapRepository
from app.services.llm_service import LLMService
from app.services.query_rewriter import QueryRewriter
from app.services.reranker import Reranker
from app.services.redis_memory import RedisMemory
from app.services.retrieval_service import RetrievalService
from app.services.prompt_builder import build_messages

logger = logging.getLogger(__name__)

_NO_INFO_MARKER = "[SIN_INFO]"
_CHITCHAT_MARKER = "[CHARLA]"
_CLARIFY_MARKER = "[ACLARAR]"
# Cuántos chars hay que bufferear en streaming antes de poder decidir el marcador.
_MAX_MARKER_LEN = max(len(_NO_INFO_MARKER), len(_CHITCHAT_MARKER), len(_CLARIFY_MARKER))


_FOLLOWUP_PROMPT = """Basándote en la conversación y en los temas disponibles en la documentación, \
generá 1 pregunta de seguimiento que el usuario probablemente quiera hacer a continuación.

REGLAS:
- Corta y en primera persona (como la escribiría el usuario).
- Que se pueda responder con los temas documentados listados abajo.
- No repitas la pregunta original ni inventes temas fuera de los listados.

Pregunta del usuario: {question}
Respuesta dada: {answer}

Temas documentados disponibles:
{topics}

Respondé ÚNICAMENTE con un JSON array de un solo string. Ejemplo: ["¿Cómo configuro X?"]"""


_TITLE_PROMPT = """Generá un título corto (máximo 5 palabras) que resuma el tema de esta \
conversación de soporte técnico. Sin comillas, sin punto final, en el idioma del usuario.

Usuario: {question}
Asistente: {answer}

Respondé ÚNICAMENTE con el título."""


_CLARIFICATION_PROMPT = """Analizá la respuesta de un asistente y detectá si le está pidiendo al \
usuario que ACLARE a qué se refiere, porque su consulta era ambigua y podría apuntar a varios \
servicios o temas distintos.

Si es una pregunta de aclaración que ofrece varias opciones, devolvé cada opción como {{"label", "query"}}:
- "label": cómo lo diría el usuario, MUY corto (1-3 palabras), nombrando el SISTEMA o SERVICIO. \
Ejemplos BUENOS: "Wi-Fi", "Office 365", "VPN", "Guaraní".
- "query": la consulta reformulada y autocontenida para esa opción, en el idioma del usuario.

Si la respuesta NO es una aclaración (responde directamente, saluda, o dice que no tiene \
información), devolvé [].

Respuesta del asistente:
{answer}

Respondé ÚNICAMENTE con un JSON array. Si es aclaración, ejemplo: \
[{{"label": "Wi-Fi", "query": "cómo me conecto al Wi-Fi de la facultad"}}]. Si no lo es: []"""


def _parse_markers(text: str) -> tuple[str, bool, bool, bool]:
    """Returns (clean_text, no_info, chitchat, clarify) según el marcador inicial."""
    stripped = text.lstrip()
    if stripped.startswith(_NO_INFO_MARKER):
        return stripped[len(_NO_INFO_MARKER):].lstrip(), True, False, False
    if stripped.startswith(_CHITCHAT_MARKER):
        return stripped[len(_CHITCHAT_MARKER):].lstrip(), False, True, False
    if stripped.startswith(_CLARIFY_MARKER):
        return stripped[len(_CLARIFY_MARKER):].lstrip(), False, False, True
    return text, False, False, False


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
                reranker_url=settings.reranker_url,
                top_k=settings.reranker_top_k,
                timeout=settings.reranker_timeout,
            )
            if settings.reranker_enabled
            else None
        )

    async def chat(self, conversation_id: str, question: str, username: str | None = None) -> ChatResponse:
        # 1. Retrieve conversation history
        history = await self._memory.get_history(conversation_id)
        is_first_turn = len(history.messages) == 0

        # 2. Query rewriting: remove implicit references to conversation history
        standalone_query = question
        if self._settings.query_rewrite_enabled:
            standalone_query = await self._rewriter.rewrite_standalone(
                question=question,
                history=history.messages[-10:],
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

        # 6. Build messages
        messages = build_messages(
            question=question,
            chunks=chunks,
            history=history.messages[-10:],
            max_context_chars=self._settings.max_context_chars,
            max_context_tokens=self._settings.max_context_tokens,
        )
        logger.info("Final answer messages built (%d turns)", len(messages))

        # 7. Call LLM
        raw_answer = await self._llm.generate(messages, log_request=True)
        answer, no_info, chitchat, clarify = _parse_markers(raw_answer)

        # 8. Collect unique sources — solo si la respuesta se apoya en la documentación
        #    (un saludo/charla, un "sin info" o una aclaración no debe mostrar fuentes)
        grounded = not no_info and not chitchat and not clarify
        sources = list(dict.fromkeys(c.source for c in chunks)) if grounded else []

        # 9. Persist turn to Redis
        await self._memory.add_turn(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            username=username,
            sources=sources,
            no_info=no_info,
        )

        if is_first_turn:
            title = await self.generate_title(question, answer)
            if title:
                await self._memory.set_title(conversation_id, title)

        return ChatResponse(answer=answer, sources=sources, no_info=no_info)

    async def chat_stream(
        self, conversation_id: str, question: str, username: str | None = None,
        regenerate: bool = False, gap_repo: GapRepository | None = None,
    ) -> AsyncIterator[str]:
        """Same pipeline as chat() but streams the LLM response token by token via SSE."""
        try:
            if regenerate:
                # Descartamos el último turno; la pregunta a regenerar es la de ese turno.
                popped = await self._memory.pop_last_turn(conversation_id)
                if popped:
                    question = popped

            history = await self._memory.get_history(conversation_id)
            is_first_turn = len(history.messages) == 0

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

            messages = build_messages(
                question=question,
                chunks=chunks,
                history=history.messages[-10:],
                max_context_chars=self._settings.max_context_chars,
                max_context_tokens=self._settings.max_context_tokens,
            )

            full_answer = ""
            buffer = ""
            marker_checked = False

            async for token in self._llm.generate_stream(messages, log_request=True):
                full_answer += token
                if not marker_checked:
                    buffer += token
                    if len(buffer) >= _MAX_MARKER_LEN:
                        marker_checked = True
                        buffer, _, _, _ = _parse_markers(buffer)
                        if buffer:
                            yield f"data: {json.dumps({'token': buffer})}\n\n"
                else:
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # Flush buffer if stream ended before we had enough chars to check
            if not marker_checked and buffer:
                buffer, _, _, _ = _parse_markers(buffer)
                if buffer:
                    yield f"data: {json.dumps({'token': buffer})}\n\n"

            clean_answer, no_info, chitchat, clarify = _parse_markers(full_answer)
            grounded = not no_info and not chitchat and not clarify
            sources = list(dict.fromkeys(c.source for c in chunks)) if grounded else []
            await self._memory.add_turn(
                conversation_id=conversation_id,
                question=question,
                answer=clean_answer,
                username=username,
                sources=sources,
                no_info=no_info,
            )

            yield f"data: {json.dumps({'done': True, 'sources': sources, 'no_info': no_info})}\n\n"

            if clarify:
                alternatives = await self.detect_clarification(clean_answer)
                if alternatives:
                    yield f"data: {json.dumps({'alternatives': alternatives})}\n\n"
            elif grounded:
                suggestions = await self.generate_followups(question, clean_answer, chunks)
                if suggestions:
                    yield f"data: {json.dumps({'suggestions': suggestions})}\n\n"

            # Título automático: solo en el primer turno de la conversación.
            if is_first_turn:
                title = await self.generate_title(question, clean_answer)
                if title:
                    await self._memory.set_title(conversation_id, title)
                    yield f"data: {json.dumps({'title': title})}\n\n"

            # Gap de documentación: si la respuesta fue [SIN_INFO], la consulta es una
            # pregunta real no cubierta. Se persiste al final (no agrega latencia a los
            # eventos del usuario) y nunca rompe el stream. No se registra en regenerate.
            if no_info and not regenerate and gap_repo is not None:
                try:
                    await gap_repo.insert(
                        conversation_id=conversation_id,
                        username=username,
                        question=question,
                        standalone_query=standalone_query,
                        model=self._settings.ollama_model,
                    )
                except Exception:
                    logger.warning("No se pudo persistir el gap de documentación", exc_info=True)

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

    async def generate_title(self, question: str, answer: str) -> str:
        """Genera un título corto y limpio para la conversación. '' ante cualquier problema."""
        prompt = _TITLE_PROMPT.format(question=question[:500], answer=answer[:500])
        try:
            raw = await self._llm.generate(
                [{"role": "user", "content": prompt}],
                options=self._llm.internal_options,
            )
            title = raw.strip().strip('"').strip("'").splitlines()[0].strip()
            return title[:80]
        except Exception:
            logger.warning("Title generation failed", exc_info=True)
            return ""

    async def generate_followups(
        self,
        question: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[str]:
        """Genera 2-3 preguntas de seguimiento basadas en la respuesta y los temas documentados.
        Devuelve [] ante cualquier problema (fallback silencioso)."""
        titles = list(dict.fromkeys(c.title for c in chunks if c.title))[:5]
        if not titles:
            return []

        prompt = _FOLLOWUP_PROMPT.format(
            question=question,
            answer=answer[:1500],
            topics="\n".join(f"- {t}" for t in titles),
        )
        try:
            raw = await self._llm.generate(
                [{"role": "user", "content": prompt}],
                options=self._llm.internal_options,
            )
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return []
            suggestions = json.loads(raw[start:end])
            cleaned = [
                s.strip()
                for s in suggestions
                if isinstance(s, str) and s.strip() and s.strip() != question
            ]
            return cleaned[:1]
        except Exception:
            logger.warning("Follow-up generation failed", exc_info=True)
            return []

    async def detect_clarification(self, answer: str) -> list[dict]:
        """Si la respuesta del modelo fue una pregunta de aclaración que ofrece varias opciones,
        las extrae como {label, query} para mostrarlas como chips. [] si la respuesta fue directa
        o ante cualquier error (fallback silencioso). Corre post-respuesta, así que no penaliza el
        time-to-first-token."""
        prompt = _CLARIFICATION_PROMPT.format(answer=answer[:1500])
        try:
            raw = await self._llm.generate(
                [{"role": "user", "content": prompt}],
                options=self._llm.internal_options,
            )
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return []
            parsed = json.loads(raw[start:end])
            result: list[dict] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "")).strip()
                query = str(item.get("query", "")).strip()
                if label and query:
                    result.append({"label": label, "query": query})
            return result[: self._settings.max_clarification_options]
        except Exception:
            logger.warning("Clarification detection failed", exc_info=True)
            return []

    async def generate_ticket_description(self, conversation_id: str) -> str:
        history = await self._memory.get_history(conversation_id)
        messages = history.messages[-20:]

        conversation_text = "\n".join(
            f"{'Usuario' if m.role == 'user' else 'Asistente'}: {m.content}"
            for m in messages
        )

        prompt = (
            "Eres un asistente experto en creación de tickets de soporte.\n\n"
            "TAREA: Analiza la conversación y genera un ticket estructurado.\n\n"

            "REGLAS ESTRICTAS:\n"
            "1. NO inventes información. Solo usa lo explícitamente mencionado.\n"
            "2. Identifica el PROBLEMA PRINCIPAL (descarta troubleshooting tangencial).\n"
            "3. Si el usuario no probó soluciones, escribe \"No documentado\".\n"
            "4. Categoría: Elige UNA sola de [Bug, Configuración, Acceso, VPN, Rendimiento, Otro].\n"
            "5. SOLO TEXTO PLANO. Sin markdown, asteriscos, negritas.\n\n"

            "FORMATO (línea vacía entre secciones):\n"
            "Problema Principal:\n"
            "[Resumen de 2-3 líneas: qué intenta el usuario, qué falla]\n\n"

            "Pasos Reproducidos:\n"
            "[Si los hay, lista clara. Si no, escribe \"No documentado\"]\n\n"

            "Intentos de Resolución:\n"
            "[Qué probó. Si nada, escribe \"No documentado\"]\n\n"

            "Comportamiento Esperado vs Real:\n"
            "[Qué debería pasar vs qué pasó]\n\n"

            "Categoría:\n"
            "[Una sola: Bug | Configuración | Acceso | VPN | Rendimiento | Otro]\n\n"

            "EJEMPLO:\n"
            "---\n"
            "Problema Principal:\n"
            "Usuario no puede acceder a VPN desde casa. Error \"Connection refused\".\n\n"
            "Pasos Reproducidos:\n"
            "1. Abre Cisco AnyConnect\n"
            "2. Ingresa credenciales\n"
            "3. Error al conectar\n\n"
            "Intentos de Resolución:\n"
            "Reinició la computadora. No probó desinstalar/reinstalar VPN.\n\n"
            "Comportamiento Esperado vs Real:\n"
            "Esperado: Conectarse y acceder a recursos internos.\n"
            "Real: Error de conexión inmediato.\n\n"
            "Categoría:\n"
            "VPN\n"
            "---\n\n"

            f"CONVERSACIÓN A ANALIZAR:\n{conversation_text}"
        )

        raw = await self._llm.generate([{"role": "user", "content": prompt}])
        return self._parse_ticket_response(raw)

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

    def _parse_ticket_response(self, text: str) -> str:
        """Valida estructura de ticket y retorna formato uniforme."""
        sections = {}
        current_section = None

        for line in text.split("\n"):
            if line.strip().endswith(":"):
                current_section = line.strip()[:-1]
                sections[current_section] = ""
            elif current_section:
                sections[current_section] += line + "\n"

        categoria = sections.get("Categoría", "").strip()
        valid_categorias = ["Bug", "Configuración", "Acceso", "VPN", "Rendimiento", "Otro"]

        if categoria not in valid_categorias:
            categoria = "Otro"

        formatted = (
            f"Problema Principal:\n{sections.get('Problema Principal', '').strip()}\n\n"
            f"Pasos Reproducidos:\n{sections.get('Pasos Reproducidos', '').strip()}\n\n"
            f"Intentos de Resolución:\n{sections.get('Intentos de Resolución', '').strip()}\n\n"
            f"Comportamiento Esperado vs Real:\n{sections.get('Comportamiento Esperado vs Real', '').strip()}\n\n"
            f"Categoría:\n{categoria}"
        )

        return formatted
