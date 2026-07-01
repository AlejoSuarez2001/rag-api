"""Modelos SQLModel — la fuente de verdad del esquema de la base.

Cada clase mapea 1:1 una tabla en el schema `analytics`. Estos modelos NO se usan
para consultar (eso sigue en asyncpg crudo en los repositorios); existen para que
Alembic pueda versionar el esquema y autogenerar migraciones a partir de su diff.

Para evolucionar el esquema: cambiá el modelo acá y corré
`alembic revision --autogenerate -m "descripcion"` seguido de `alembic upgrade head`.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Boolean, Column, Index, SmallInteger, Text, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP
from sqlmodel import Field, SQLModel

from app.db import SCHEMA


class Feedback(SQLModel, table=True):
    """Feedback de usuarios sobre respuestas del asistente.

    Snapshot completo (pregunta + respuesta + fuentes) porque el historial de Redis
    expira a los 7 días y el reporte debe sobrevivir para análisis posterior.
    """

    __tablename__ = "feedback"
    __table_args__ = (
        Index("idx_feedback_created", "created_at"),
        Index("idx_feedback_rating", "rating"),
        {"schema": SCHEMA},
    )

    id: Optional[int] = Field(
        default=None,
        sa_column=Column(BigInteger, primary_key=True, autoincrement=True),
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")),
    )
    conversation_id: str = Field(sa_column=Column(Text, nullable=False))
    username: Optional[str] = Field(default=None, sa_column=Column(Text))
    rating: int = Field(sa_column=Column(SmallInteger, nullable=False))
    categories: list[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(Text), nullable=False, server_default=text("'{}'")),
    )
    comment: Optional[str] = Field(default=None, sa_column=Column(Text))
    question: str = Field(sa_column=Column(Text, nullable=False))
    answer: str = Field(sa_column=Column(Text, nullable=False))
    sources: list = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    )
    no_info: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default=text("false")),
    )
    model: Optional[str] = Field(default=None, sa_column=Column(Text))


class DocGap(SQLModel, table=True):
    """Gap de documentación: una consulta real que disparó [SIN_INFO].

    La base de conocimiento no la cubre. Se guarda acá (Redis expira) para que los
    admins puedan priorizar qué documentar.
    """

    __tablename__ = "doc_gaps"
    __table_args__ = (
        Index("idx_doc_gaps_created", "created_at"),
        {"schema": SCHEMA},
    )

    id: Optional[int] = Field(
        default=None,
        sa_column=Column(BigInteger, primary_key=True, autoincrement=True),
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")),
    )
    conversation_id: str = Field(sa_column=Column(Text, nullable=False))
    username: Optional[str] = Field(default=None, sa_column=Column(Text))
    question: str = Field(sa_column=Column(Text, nullable=False))
    standalone_query: str = Field(sa_column=Column(Text, nullable=False))
    model: Optional[str] = Field(default=None, sa_column=Column(Text))
