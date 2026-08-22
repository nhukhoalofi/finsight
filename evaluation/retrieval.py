"""Retriever-agnostic chunk-level retrieval evaluation."""

import statistics
import time
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from app.retrieval.types import RetrievalHit
from evaluation.datasets.retrieval import RetrievalEvalCase
from evaluation.metrics.retrieval import recall_at_k, reciprocal_rank


class RetrievalSearcher(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]: ...


class RetrievalQueryResult(BaseModel):
    case_id: str
    question: str
    relevant_chunk_ids: list[str]
    hits: list[RetrievalHit]
    first_relevant_rank: int | None
    recall_at_5: float
    recall_at_10: float
    mrr: float
    latency_ms: float


class RetrievalEvaluationSummary(BaseModel):
    case_count: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float


class RetrievalEvaluation(BaseModel):
    summary: RetrievalEvaluationSummary
    results: list[RetrievalQueryResult] = Field(default_factory=list)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def evaluate_retriever(
    cases: Sequence[RetrievalEvalCase],
    retriever: RetrievalSearcher,
    *,
    top_k: int = 10,
) -> RetrievalEvaluation:
    """Evaluate a public retriever interface using question text only."""
    if top_k < 10:
        raise ValueError("top_k must be >= 10 because evaluation reports Recall@10")
    results: list[RetrievalQueryResult] = []
    for case in cases:
        started = time.perf_counter()
        hits = retriever.search(case.question, top_k)
        latency_ms = (time.perf_counter() - started) * 1_000
        relevant = set(case.relevant_chunk_ids)
        results.append(
            RetrievalQueryResult(
                case_id=case.id,
                question=case.question,
                relevant_chunk_ids=case.relevant_chunk_ids,
                hits=hits,
                first_relevant_rank=next((hit.rank for hit in hits if hit.chunk_id in relevant), None),
                recall_at_5=recall_at_k(hits, relevant, 5),
                recall_at_10=recall_at_k(hits, relevant, 10),
                mrr=reciprocal_rank(hits, relevant),
                latency_ms=latency_ms,
            )
        )
    latencies = [result.latency_ms for result in results]
    count = len(results)
    return RetrievalEvaluation(
        summary=RetrievalEvaluationSummary(
            case_count=count,
            recall_at_5=sum(result.recall_at_5 for result in results) / count if count else 0.0,
            recall_at_10=sum(result.recall_at_10 for result in results) / count if count else 0.0,
            mrr=sum(result.mrr for result in results) / count if count else 0.0,
            mean_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
            p50_latency_ms=_percentile(latencies, 0.5),
            p95_latency_ms=_percentile(latencies, 0.95),
        ),
        results=results,
    )
