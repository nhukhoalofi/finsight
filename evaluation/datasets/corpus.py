"""Frozen processed-corpus definitions for reproducible retrieval evaluation."""

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.ingestion.metadata import DocumentChunk
from app.retrieval.corpus import load_processed_chunks
from evaluation.datasets.retrieval import RetrievalEvalCase


class ChunkingDefinition(BaseModel):
    target_size: int
    overlap: int
    hard_limit: int


class RetrievalCorpusManifest(BaseModel):
    schema_version: str
    name: str
    chunking: ChunkingDefinition
    source_documents: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_documents(self) -> "RetrievalCorpusManifest":
        if len(self.source_documents) != len(set(self.source_documents)):
            raise ValueError("corpus manifest source_documents must be unique")
        return self


def load_retrieval_corpus_manifest(path: Path) -> RetrievalCorpusManifest:
    try:
        return RetrievalCorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ValueError(f"Invalid retrieval corpus manifest: {path}") from error


def load_frozen_corpus_chunks(
    processed_path: Path, manifest: RetrievalCorpusManifest
) -> list[DocumentChunk]:
    """Return only manifest-listed documents, in manifest and JSONL order."""
    by_source: dict[str, list[DocumentChunk]] = {}
    for chunk in load_processed_chunks(processed_path):
        by_source.setdefault(Path(chunk.source_name).stem, []).append(chunk)

    missing = [source for source in manifest.source_documents if source not in by_source]
    if missing:
        raise ValueError("Missing processed corpus documents: " + ", ".join(missing))
    return [chunk for source in manifest.source_documents for chunk in by_source[source]]


def validate_cases_reference_corpus(
    cases: list[RetrievalEvalCase], manifest: RetrievalCorpusManifest
) -> None:
    available = set(manifest.source_documents)
    missing = sorted({case.source_name for case in cases} - available)
    if missing:
        raise ValueError("Golden cases absent from corpus manifest: " + ", ".join(missing))


def validate_cases_reference_chunks(
    cases: list[RetrievalEvalCase], chunks: list[DocumentChunk]
) -> None:
    """Ensure frozen golden relevance IDs exist and belong to their source document."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for case in cases:
        missing = sorted(chunk_id for chunk_id in case.relevant_chunk_ids if chunk_id not in chunks_by_id)
        if missing:
            raise ValueError(f"Missing relevant chunk ids for {case.id}: " + ", ".join(missing))
        for chunk_id in case.relevant_chunk_ids:
            actual_source = Path(chunks_by_id[chunk_id].source_name).stem
            if actual_source != case.source_name:
                raise ValueError(
                    f"Relevant chunk {chunk_id} belongs to {actual_source} "
                    f"but expected {case.source_name}"
                )
