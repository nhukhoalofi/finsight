from typing import Any

from pydantic import BaseModel, Field


class RetrievalHit(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    source_name: str
    page: int | None
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)
