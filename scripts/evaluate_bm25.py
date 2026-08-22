import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval.sparse import BM25Retriever
from evaluation.datasets.corpus import (
    load_frozen_corpus_chunks,
    load_retrieval_corpus_manifest,
    validate_cases_reference_chunks,
    validate_cases_reference_corpus,
)
from evaluation.datasets.retrieval import load_retrieval_eval_cases
from evaluation.reporting import build_retrieval_report, write_retrieval_report
from evaluation.retrieval import evaluate_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the BM25 retriever on frozen retrieval gold.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/financebench_retrieval_dev.jsonl"),
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/documents"))
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("evaluation/datasets/financebench_retrieval_dev_corpus.json"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args()

    cases = load_retrieval_eval_cases(args.dataset)
    corpus_manifest = load_retrieval_corpus_manifest(args.corpus_manifest)
    validate_cases_reference_corpus(cases, corpus_manifest)
    chunks = load_frozen_corpus_chunks(args.corpus, corpus_manifest)
    validate_cases_reference_chunks(cases, chunks)
    retriever = BM25Retriever(chunks)
    evaluation = evaluate_retriever(cases, retriever, top_k=args.top_k)
    report = build_retrieval_report(
        retriever_name="bm25",
        dataset_path=args.dataset,
        corpus_manifest_path=args.corpus_manifest,
        corpus_document_count=len({chunk.document_id for chunk in chunks}),
        corpus_chunk_count=len(chunks),
        retriever_parameters={"k1": retriever.k1, "b": retriever.b},
        top_k=args.top_k,
        evaluation=evaluation,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"bm25_{timestamp}.json"
    write_retrieval_report(report, output)
    summary = evaluation.summary
    print(f"Evaluated queries: {summary.case_count}")
    print(f"Recall@5:  {summary.recall_at_5:.4f}")
    print(f"Recall@10: {summary.recall_at_10:.4f}")
    print(f"MRR:       {summary.mrr:.4f}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
