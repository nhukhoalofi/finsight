from fastapi import FastAPI

from app.api.routes.health import router as health_router

app = FastAPI(
    title="FinSight API",
    version="0.1.0",
    description="Multi-Agent Financial Intelligence RAG",
)

app.include_router(health_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "FinSight API",
        "status": "running",
    }
