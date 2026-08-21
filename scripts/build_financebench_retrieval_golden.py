"""Build the fixed FinanceBench chunk-level retrieval development dataset."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.retrieval.corpus import load_processed_chunks
from evaluation.datasets.retrieval import RetrievalEvalCase
from evaluation.financebench import load_cases, map_case_to_relevant_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/documents"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/datasets/financebench_retrieval_dev.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    root = get_settings().financebench_path
    chunks = load_processed_chunks(args.corpus)
    selected: list[RetrievalEvalCase] = []
    for case in load_cases(root / "data" / "financebench_open_source.jsonl"):
        if not (root / "pdfs" / f"{case.source_name}.pdf").is_file():
            continue
        relevant_chunk_ids = sorted(map_case_to_relevant_chunks(case, chunks))
        if not relevant_chunk_ids:
            continue
        selected.append(
            RetrievalEvalCase(
                id=case.question_id,
                question=case.question,
                source_name=case.source_name,
                evidence_pages=sorted(case.evidence_pages),
                relevant_chunk_ids=relevant_chunk_ids,
            )
        )
        if len(selected) == args.limit:
            break
    if len(selected) != args.limit:
        raise SystemExit(f"Requested {args.limit} cases but only found {len(selected)} valid cases")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case.model_dump(), sort_keys=True) + "\n" for case in selected),
        encoding="utf-8",
    )
    documents = sorted({case.source_name for case in selected})
    print(f"Wrote {len(selected)} cases across {len(documents)} documents to {args.output}")
    print("Documents: " + ", ".join(documents))


if __name__ == "__main__":
    main()
