from dataclasses import dataclass
from typing import Any

from app.ingestion.metadata import (
    DocumentChunk,
    ParsedPage,
    build_chunk_id,
    content_hash,
)


@dataclass(frozen=True)
class ChunkingConfig:
    target_size: int = 500
    overlap: int = 64
    hard_limit: int = 600


DEFAULT_CHUNKING_CONFIG = ChunkingConfig()


def chunk_pages(
    pages: list[ParsedPage],
    document_id: str,
    source_name: str,
    metadata: dict[str, Any],
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> list[DocumentChunk]:
    if config.target_size <= 0:
        raise ValueError("target_size must be positive")
    if config.overlap < 0:
        raise ValueError("overlap must be non-negative")
    if config.overlap >= config.target_size:
        raise ValueError("overlap must be smaller than target_size")
    if config.hard_limit < config.target_size:
        raise ValueError("hard_limit must be >= target_size")

    step = config.target_size - config.overlap
    chunks: list[DocumentChunk] = []
    for page in pages:
        words = page.text.split()
        start = 0
        while start < len(words):
            part = words[start : start + config.target_size]
            text = " ".join(part).strip()
            if text:
                ordinal = len(chunks)
                chunks.append(DocumentChunk(
                    chunk_id=build_chunk_id(document_id, page.page_number, ordinal, text),
                    document_id=document_id, text=text, source_name=source_name,
                    page=page.page_number, ordinal=ordinal, content_hash=content_hash(text), metadata=metadata,
                ))
            if start + config.target_size >= len(words):
                break
            start += step
    return chunks
