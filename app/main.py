from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Multi-Agent Financial Intelligence RAG",
)

app.include_router(health_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": f"{settings.app_name} API",
        "status": "running",
    }
