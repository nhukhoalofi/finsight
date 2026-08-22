from pathlib import Path

import pytest

from app.retrieval.types import RetrievalHit
from evaluation.datasets.retrieval import RetrievalEvalCase
from evaluation.reporting import build_retrieval_report, write_retrieval_report
from evaluation.retrieval import evaluate_retriever


def hit(chunk_id: str, rank: int) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id="document",
        text="text",
        source_name="source.pdf",
        page=1,
        score=float(11 - rank),
        rank=rank,
    )


class FakeRetriever:
    def __init__(self, rankings: dict[str, list[RetrievalHit]]) -> None:
        self.rankings = rankings
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        self.queries.append(query)
        return self.rankings[query][:top_k]


def case(identifier: str, relevant: list[str]) -> RetrievalEvalCase:
    return RetrievalEvalCase(
        id=identifier,
        question=identifier,
        source_name="source",
        evidence_pages=[1],
        relevant_chunk_ids=relevant,
    )


def test_evaluator_metrics_and_ordering() -> None:
    cases = [case("perfect", ["a"]), case("late", ["b", "c"]), case("miss", ["d"])]
    rankings = {
        "perfect": [hit("a", 1)],
        "late": [
            hit("x", 1),
            hit("y", 2),
            hit("z", 3),
            hit("w", 4),
            hit("v", 5),
            hit("b", 6),
            hit("c", 8),
        ],
        "miss": [hit("x", 1)],
    }
    retriever = FakeRetriever(rankings)
    evaluation = evaluate_retriever(cases, retriever)
    assert retriever.queries == ["perfect", "late", "miss"]
    assert [result.case_id for result in evaluation.results] == ["perfect", "late", "miss"]
    assert evaluation.results[0].mrr == 1.0
    assert evaluation.results[1].first_relevant_rank == 6
    assert evaluation.results[1].recall_at_5 == 0.0
    assert evaluation.results[1].recall_at_10 == 1.0
    assert evaluation.results[2].mrr == 0.0
    assert evaluation.summary.recall_at_5 == 1 / 3
    assert evaluation.summary.recall_at_10 == 2 / 3
    assert evaluation.summary.mrr == (1 + 1 / 6) / 3


def test_evaluator_rejects_top_k_below_ten() -> None:
    with pytest.raises(ValueError, match="top_k must be >= 10 because evaluation reports Recall@10"):
        evaluate_retriever([case("q", ["a"])], FakeRetriever({"q": [hit("a", 1)]}), top_k=5)


def test_report_serialization_contains_identity_rankings_and_parameters(tmp_path: Path) -> None:
    dataset = tmp_path / "golden.jsonl"
    dataset.write_text('{"id":"q"}\n', encoding="utf-8")
    evaluation = evaluate_retriever([case("q", ["a"])], FakeRetriever({"q": [hit("a", 1)]}))
    report = build_retrieval_report(
        retriever_name="fake",
        dataset_path=dataset,
        corpus_manifest_path=dataset,
        corpus_document_count=1,
        corpus_chunk_count=1,
        retriever_parameters={"example": True},
        top_k=10,
        evaluation=evaluation,
    )
    output = tmp_path / "report.json"
    write_retrieval_report(report, output)
    contents = output.read_text(encoding="utf-8")
    assert '"dataset_sha256"' in contents
    assert '"corpus_manifest_sha256"' in contents
    assert '"retriever_parameters"' in contents
    assert '"hits"' in contents
