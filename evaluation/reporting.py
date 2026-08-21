"""Reproducible JSON report model and writer for retrieval evaluations."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evaluation.retrieval import RetrievalEvaluation


class RetrievalEvaluationReport(BaseModel):
    schema_version: str = "1.0"
    suite: str = "retrieval"
    retriever_name: str
    dataset_path: str
    dataset_sha256: str
    dataset_case_count: int
    corpus_manifest_path: str
    corpus_manifest_sha256: str
    corpus_document_count: int
    corpus_chunk_count: int
    retriever_parameters: dict[str, Any]
    top_k: int
    metric_definition: str = "chunk_recall_at_k_and_mrr_v1"
    created_at: datetime
    evaluation: RetrievalEvaluation


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_retrieval_report(
    *,
    retriever_name: str,
    dataset_path: Path,
    corpus_manifest_path: Path,
    corpus_document_count: int,
    corpus_chunk_count: int,
    retriever_parameters: dict[str, Any],
    top_k: int,
    evaluation: RetrievalEvaluation,
) -> RetrievalEvaluationReport:
    return RetrievalEvaluationReport(
        retriever_name=retriever_name,
        dataset_path=dataset_path.as_posix(),
        dataset_sha256=file_sha256(dataset_path),
        dataset_case_count=evaluation.summary.case_count,
        corpus_manifest_path=corpus_manifest_path.as_posix(),
        corpus_manifest_sha256=file_sha256(corpus_manifest_path),
        corpus_document_count=corpus_document_count,
        corpus_chunk_count=corpus_chunk_count,
        retriever_parameters=retriever_parameters,
        top_k=top_k,
        created_at=datetime.now(UTC),
        evaluation=evaluation,
    )


def write_retrieval_report(report: RetrievalEvaluationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
