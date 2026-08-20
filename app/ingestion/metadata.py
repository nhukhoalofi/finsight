import hashlib
from typing import Any

from pydantic import BaseModel, Field


class ParsedPage(BaseModel):
    page_number: int
    text: str


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    source_name: str
    page: int | None = None
    section: str | None = None
    ordinal: int
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def normalized_text(text: str) -> str:
    return " ".join(text.split())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_chunk_id(document_id: str, page: int | None, ordinal: int, text: str) -> str:
    payload = f"{document_id}:{page}:{ordinal}:{normalized_text(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
