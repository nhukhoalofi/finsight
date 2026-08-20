import json
from pathlib import Path

from app.ingestion.metadata import DocumentChunk
from evaluation.financebench import (
    FinanceBenchCase,
    load_cases,
    map_case_to_relevant_chunks,
    normalize_document_name,
    to_pypdf_page,
)


def chunk(chunk_id: str, source: str, page: int) -> DocumentChunk:
    return DocumentChunk(chunk_id=chunk_id, document_id="doc", text="text", source_name=source, page=page, ordinal=0, content_hash=chunk_id)


def test_document_page_mapping_and_normalization() -> None:
    case = FinanceBenchCase(question_id="q", question="question", source_name="Source_A", evidence_pages={10, 11})
    chunks = [chunk("one", "source_a.pdf", 10), chunk("two", "SOURCE_A.pdf", 10), chunk("three", "source_a.pdf", 11), chunk("wrong", "source_b.pdf", 10)]
    assert normalize_document_name("Source_A.pdf") == normalize_document_name("source_a")
    assert map_case_to_relevant_chunks(case, chunks) == {"one", "two", "three"}


def test_missing_evidence_is_not_evaluable() -> None:
    case = FinanceBenchCase(question_id="q", question="question", source_name="missing", evidence_pages={1})
    assert map_case_to_relevant_chunks(case, [chunk("one", "source.pdf", 1)]) == set()


def test_financebench_pages_are_converted_to_pypdf_pages(tmp_path: Path) -> None:
    source = tmp_path / "cases.jsonl"
    source.write_text(
        json.dumps(
            {
                "financebench_id": "q",
                "question": "question",
                "doc_name": "source",
                "evidence": [{"evidence_page_num": 59}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert to_pypdf_page(59) == 60
    assert load_cases(source)[0].evidence_pages == {60}
