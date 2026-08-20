
from qdrant_client import AsyncQdrantClient

from app.config import get_settings


def get_qdrant_client() -> AsyncQdrantClient:
    """Create an async Qdrant client for infrastructure checks."""
    return AsyncQdrantClient(url=get_settings().qdrant_url)


async def check_qdrant_connection() -> None:
    """Raise if Qdrant cannot serve a lightweight collections request."""
    client = get_qdrant_client()
    try:
        await client.get_collections()
    finally:
        await client.close()
