from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=128, description="Unique conversation identifier")
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    regenerate: bool = Field(False, description="Si true, descarta la última respuesta del historial y la vuelve a generar")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    no_info: bool = False


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    sources: list[str] = Field(default_factory=list)
    no_info: bool = False
    rating: Optional[int] = None


class ConversationHistory(BaseModel):
    conversation_id: str
    username: Optional[str] = None
    title: Optional[str] = None
    messages: list[Message] = Field(default_factory=list)
    updated_at: Optional[str] = None


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str | None = None
    preview: str | None = None
    search_text: str | None = None
    updated_at: str | None = None

class ConversationListResponse(BaseModel):
    username: str
    conversations: list[ConversationSummary]


class SharedConversation(BaseModel):
    share_token: str
    messages: list[Message]
    created_at: str


class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float
    chunk_id: Optional[str] = None
    title: Optional[str] = None
    position: Optional[int] = None


# ---------------------------------------------------------------------------
# Feedback de usuarios sobre respuestas del asistente
# ---------------------------------------------------------------------------

class FeedbackCategory(str, Enum):
    """Taxonomía de problemas comunes en respuestas RAG.

    Permite distinguir fallos de recuperación (retrieval) de fallos de
    generación al analizar el feedback agregado.
    """
    INFO_INCORRECTA = "info_incorrecta"        # alucinación / dato inventado
    NO_ENCONTRO = "no_encontro"                # info existe pero no la recuperó
    IRRELEVANTE = "irrelevante"                # no responde lo preguntado
    FUENTE_INCORRECTA = "fuente_incorrecta"    # cita mal / fuente ruidosa
    DESACTUALIZADA = "desactualizada"          # doc viejo en el índice
    FORMATO = "formato"                        # formato o idioma
    OTRO = "otro"


class FeedbackCreate(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=128)
    rating: int = Field(..., ge=-1, le=1, description="1 = positivo (👍), -1 = negativo (👎)")
    categories: list[FeedbackCategory] = Field(default_factory=list)
    comment: Optional[str] = Field(None, max_length=2000)
    # Snapshot del contexto: el historial de Redis expira, esto no.
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list)
    no_info: bool = False


class FeedbackRecord(BaseModel):
    id: int
    created_at: str
    conversation_id: str
    username: Optional[str] = None
    rating: int
    categories: list[str] = Field(default_factory=list)
    comment: Optional[str] = None
    question: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    no_info: bool = False
    model: Optional[str] = None


class FeedbackCreatedResponse(BaseModel):
    id: int


class FeedbackListResponse(BaseModel):
    total: int
    items: list[FeedbackRecord]


class FeedbackStats(BaseModel):
    total: int
    positive: int
    negative: int
    by_category: dict[str, int]
