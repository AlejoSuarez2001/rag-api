import logging
from typing import Optional

import asyncpg

from app.config import Settings
from app.models.schemas import GapRecord, GapStats, GapTopQuery

logger = logging.getLogger(__name__)


class GapRepository:
    """Persistencia de gaps de documentación en PostgreSQL.

    Cada gap es una consulta real que disparó [SIN_INFO]: la base de conocimiento
    no la cubre. Se guarda acá (el historial de Redis expira a los 7 días) para que
    los admins puedan ver qué documentar a continuación. El esquema (tabla
    `analytics.doc_gaps`) lo gestiona Alembic; este repositorio solo consulta.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, settings: Settings) -> "GapRepository":
        pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            min_size=settings.postgres_pool_min,
            max_size=settings.postgres_pool_max,
            # Las tablas viven en el schema `analytics` (gestionado por Alembic).
            # Fijamos search_path para que las queries resuelvan sin calificar.
            server_settings={"search_path": settings.db_schema},
        )
        logger.info(
            "GapRepository conectado a Postgres %s:%s/%s",
            settings.postgres_host,
            settings.postgres_port,
            settings.postgres_db,
        )
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def insert(
        self,
        *,
        conversation_id: str,
        username: Optional[str],
        question: str,
        standalone_query: str,
        model: Optional[str],
    ) -> int:
        row = await self._pool.fetchrow(
            """
            INSERT INTO doc_gaps
                (conversation_id, username, question, standalone_query, model)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            conversation_id,
            username,
            question,
            standalone_query,
            model,
        )
        return row["id"]

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[int, list[GapRecord]]:
        total = await self._pool.fetchval("SELECT count(*) FROM doc_gaps")
        rows = await self._pool.fetch(
            """
            SELECT id, created_at, conversation_id, username, question,
                   standalone_query, model
            FROM doc_gaps
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
        return total, [self._to_record(r) for r in rows]

    async def stats(self, top_limit: int = 15) -> GapStats:
        total = await self._pool.fetchval("SELECT count(*) FROM doc_gaps")
        # Agrupación naive por texto normalizado (lowercase + trim). El accent-strip
        # con unaccent queda para v2.
        top_rows = await self._pool.fetch(
            """
            SELECT btrim(lower(standalone_query)) AS query, count(*) AS n
            FROM doc_gaps
            GROUP BY btrim(lower(standalone_query))
            ORDER BY n DESC, query
            LIMIT $1
            """,
            top_limit,
        )
        return GapStats(
            total=total or 0,
            top=[GapTopQuery(query=r["query"], count=r["n"]) for r in top_rows],
        )

    @staticmethod
    def _to_record(row: asyncpg.Record) -> GapRecord:
        return GapRecord(
            id=row["id"],
            created_at=row["created_at"].isoformat(),
            conversation_id=row["conversation_id"],
            username=row["username"],
            question=row["question"],
            standalone_query=row["standalone_query"],
            model=row["model"],
        )
