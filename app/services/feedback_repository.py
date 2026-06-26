import json
import logging
from typing import Optional

import asyncpg

from app.config import Settings
from app.models.schemas import (
    FeedbackCreate,
    FeedbackRecord,
    FeedbackStats,
)

logger = logging.getLogger(__name__)


_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS feedback (
        id              BIGSERIAL PRIMARY KEY,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        conversation_id TEXT        NOT NULL,
        username        TEXT,
        rating          SMALLINT    NOT NULL,
        categories      TEXT[]      NOT NULL DEFAULT '{}',
        comment         TEXT,
        question        TEXT        NOT NULL,
        answer          TEXT        NOT NULL,
        sources         JSONB       NOT NULL DEFAULT '[]',
        no_info         BOOLEAN     NOT NULL DEFAULT FALSE,
        model           TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback (created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_feedback_rating  ON feedback (rating);
"""


class FeedbackRepository:
    """Persistencia de feedback de usuarios en PostgreSQL.

    Mantiene un snapshot completo (pregunta, respuesta, fuentes) porque el
    historial en Redis expira a los 7 días y el reporte debe sobrevivir para
    análisis posterior.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, settings: Settings) -> "FeedbackRepository":
        pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            min_size=settings.postgres_pool_min,
            max_size=settings.postgres_pool_max,
        )
        async with pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)
        logger.info(
            "FeedbackRepository conectado a Postgres %s:%s/%s",
            settings.postgres_host,
            settings.postgres_port,
            settings.postgres_db,
        )
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def insert(
        self,
        feedback: FeedbackCreate,
        username: Optional[str],
        model: Optional[str],
    ) -> int:
        row = await self._pool.fetchrow(
            """
            INSERT INTO feedback
                (conversation_id, username, rating, categories, comment,
                 question, answer, sources, no_info, model)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            """,
            feedback.conversation_id,
            username,
            feedback.rating,
            [c.value for c in feedback.categories],
            feedback.comment,
            feedback.question,
            feedback.answer,
            json.dumps(feedback.sources),
            feedback.no_info,
            model,
        )
        return row["id"]

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        rating: Optional[int] = None,
        categories: Optional[list[str]] = None,
    ) -> tuple[int, list[FeedbackRecord]]:
        conditions: list[str] = []
        args: list = []

        if rating is not None:
            args.append(rating)
            conditions.append(f"rating = ${len(args)}")
        if categories:
            # solapamiento de arrays: la fila matchea si tiene CUALQUIERA (OR)
            args.append(categories)
            conditions.append(f"categories && ${len(args)}::text[]")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        total = await self._pool.fetchval(
            f"SELECT count(*) FROM feedback {where}", *args
        )

        args.extend([limit, offset])
        rows = await self._pool.fetch(
            f"""
            SELECT id, created_at, conversation_id, username, rating, categories,
                   comment, question, answer, sources, no_info, model
            FROM feedback
            {where}
            ORDER BY created_at DESC
            LIMIT ${len(args) - 1} OFFSET ${len(args)}
            """,
            *args,
        )
        return total, [self._to_record(r) for r in rows]

    async def stats(self) -> FeedbackStats:
        totals = await self._pool.fetchrow(
            """
            SELECT
                count(*)                                   AS total,
                count(*) FILTER (WHERE rating = 1)         AS positive,
                count(*) FILTER (WHERE rating = -1)        AS negative
            FROM feedback
            """
        )
        cat_rows = await self._pool.fetch(
            """
            SELECT category, count(*) AS n
            FROM feedback, unnest(categories) AS category
            GROUP BY category
            ORDER BY n DESC
            """
        )
        return FeedbackStats(
            total=totals["total"] or 0,
            positive=totals["positive"] or 0,
            negative=totals["negative"] or 0,
            by_category={r["category"]: r["n"] for r in cat_rows},
        )

    @staticmethod
    def _to_record(row: asyncpg.Record) -> FeedbackRecord:
        return FeedbackRecord(
            id=row["id"],
            created_at=row["created_at"].isoformat(),
            conversation_id=row["conversation_id"],
            username=row["username"],
            rating=row["rating"],
            categories=list(row["categories"] or []),
            comment=row["comment"],
            question=row["question"],
            answer=row["answer"],
            sources=json.loads(row["sources"]) if row["sources"] else [],
            no_info=row["no_info"],
            model=row["model"],
        )
