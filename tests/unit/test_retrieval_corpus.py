import json
from pathlib import Path

import pytest

from app.ingestion.metadata import DocumentChunk
from evaluation.datasets.corpus import (
    load_frozen_corpus_chunks,
    load_retrieval_corpus_manifest,
    validate_cases_reference_corpus,
)
from evaluation.datasets.retrieval import RetrievalEvalCase


def write_manifest(path: Path, sources: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "test",
                "chunking": {"target_size": 500, "overlap": 64, "hard_limit": 600},
                "source_documents": sources,
            }
        ),
        encoding="utf-8",
    )


def write_chunk(path: Path, source: str, chunk_id: str) -> None:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=source,
        text="text",
        source_name=f"{source}.pdf",
        page=1,
        ordinal=0,
        content_hash=chunk_id,
    )
    path.write_text(chunk.model_dump_json() + "\n", encoding="utf-8")


def test_frozen_corpus_uses_manifest_order_and_excludes_extra_documents(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_chunk(corpus / "a.jsonl", "alpha", "alpha-chunk")
    write_chunk(corpus / "b.jsonl", "beta", "beta-chunk")
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, ["beta", "alpha"])

    manifest = load_retrieval_corpus_manifest(manifest_path)
    chunks = load_frozen_corpus_chunks(corpus, manifest)
    assert [chunk.chunk_id for chunk in chunks] == ["beta-chunk", "alpha-chunk"]


def test_manifest_rejects_duplicate_or_missing_documents(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    write_manifest(duplicate, ["alpha", "alpha"])
    with pytest.raises(ValueError, match="Invalid retrieval corpus manifest"):
        load_retrieval_corpus_manifest(duplicate)

    missing = tmp_path / "missing.json"
    write_manifest(missing, ["missing"])
    with pytest.raises(ValueError, match="Missing processed corpus documents: missing"):
        load_frozen_corpus_chunks(tmp_path, load_retrieval_corpus_manifest(missing))


def test_golden_cases_must_reference_manifest_documents(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, ["alpha"])
    case = RetrievalEvalCase(
        id="case",
        question="question",
        source_name="beta",
        evidence_pages=[1],
        relevant_chunk_ids=["chunk"],
    )
    with pytest.raises(ValueError, match="Golden cases absent from corpus manifest: beta"):
        validate_cases_reference_corpus([case], load_retrieval_corpus_manifest(manifest_path))
