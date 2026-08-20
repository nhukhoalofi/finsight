import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.database.session import check_postgres_connection
from app.vectorstore.qdrant_store import check_qdrant_connection

router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@router.get("/live")
async def liveness() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "finsight-api",
    }


@router.get("/ready", response_model=None)
async def readiness() -> dict[str, object] | JSONResponse:
    postgres_result, qdrant_result = await asyncio.gather(
        check_postgres_connection(),
        check_qdrant_connection(),
        return_exceptions=True,
    )
    dependencies = {
        "postgres": "ok" if not isinstance(postgres_result, Exception) else "error",
        "qdrant": "ok" if not isinstance(qdrant_result, Exception) else "error",
    }

    if all(status == "ok" for status in dependencies.values()):
        return {"status": "ready", "dependencies": dependencies}

    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "dependencies": dependencies},
    )
