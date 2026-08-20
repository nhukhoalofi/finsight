
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    """Create the application's async PostgreSQL engine."""
    return create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)


async def check_postgres_connection() -> None:
    """Raise if PostgreSQL cannot execute a lightweight readiness query."""
    async with get_engine().connect() as connection:
        await connection.execute(text("SELECT 1"))
