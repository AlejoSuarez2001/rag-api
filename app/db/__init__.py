"""Capa de base de datos: modelado del esquema (SQLModel) y constantes compartidas.

Las tablas de rag-api viven en un schema de Postgres propio (`analytics`) para no
pisar las del servicio de ingesta, que comparte la misma base. Las queries en runtime
siguen usando asyncpg crudo; estos modelos son la fuente de verdad del esquema y la
base sobre la que Alembic genera las migraciones.
"""

SCHEMA = "analytics"

__all__ = ["SCHEMA"]
