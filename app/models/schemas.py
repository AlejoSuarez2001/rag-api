from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=128, description="Unique conversation identifier")
    question: str = Field(..., min_length=1, max_length=2000, description="User question")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    sources: list[str] = Field(default_factory=list)


class ConversationHistory(BaseModel):
    conversation_id: str
    username: Optional[str] = None
    messages: list[Message] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    conversation_id: str
    preview: str | None = None

class ConversationListResponse(BaseModel):
    username: str
    conversations: list[ConversationSummary]


class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float
    chunk_id: Optional[str] = None
    title: Optional[str] = None
    position: Optional[int] = None
