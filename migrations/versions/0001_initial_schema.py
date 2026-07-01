"""initial schema: analytics.feedback + analytics.doc_gaps

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-30

Crea el schema propio de rag-api y las dos tablas que antes se auto-creaban en los
repositorios (feedback, doc_gaps). De acá en más el esquema se evoluciona con nuevas
revisiones de Alembic, no con CREATE TABLE IF NOT EXISTS.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "analytics"


def upgrade() -> None:
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("username", sa.Text(), nullable=True),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("categories", postgresql.ARRAY(sa.Text()), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("sources", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("no_info", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("model", sa.Text(), nullable=True),
        schema=SCHEMA,
    )
    op.create_index("idx_feedback_created", "feedback", ["created_at"], schema=SCHEMA)
    op.create_index("idx_feedback_rating", "feedback", ["rating"], schema=SCHEMA)

    op.create_table(
        "doc_gaps",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("username", sa.Text(), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("standalone_query", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=True),
        schema=SCHEMA,
    )
    op.create_index("idx_doc_gaps_created", "doc_gaps", ["created_at"], schema=SCHEMA)


def downgrade() -> None:
    op.drop_table("doc_gaps", schema=SCHEMA)
    op.drop_table("feedback", schema=SCHEMA)
    op.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
