from pathlib import Path

from pydantic import ValidationError

from app.ingestion.metadata import DocumentChunk


def load_processed_chunks(path: Path) -> list[DocumentChunk]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    chunks: list[DocumentChunk] = []
    for file in files:
        with file.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    chunks.append(DocumentChunk.model_validate_json(line))
                except ValidationError as error:
                    raise ValueError(f"Invalid chunk JSONL at {file}:{line_number}") from error
    return chunks
