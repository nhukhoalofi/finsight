import json
from pathlib import Path

from pydantic import BaseModel

from app.ingestion.metadata import DocumentChunk

FINANCEBENCH_TO_PYPDF_PAGE_OFFSET = 1


class FinanceBenchCase(BaseModel):
    question_id: str
    question: str
    source_name: str
    evidence_pages: set[int]


def normalize_document_name(name: str) -> str:
    return Path(name).stem.replace("_", "").lower()


def to_pypdf_page(financebench_page: int) -> int:
    """Convert FinanceBench's zero-based evidence page to PyPDF's one-based page."""
    return financebench_page + FINANCEBENCH_TO_PYPDF_PAGE_OFFSET


def load_cases(path: Path) -> list[FinanceBenchCase]:
    cases: list[FinanceBenchCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        pages = {to_pypdf_page(item["evidence_page_num"]) for item in record.get("evidence", [])}
        if pages:
            cases.append(FinanceBenchCase(question_id=record["financebench_id"], question=record["question"], source_name=record["doc_name"], evidence_pages=pages))
    return cases


def map_case_to_relevant_chunks(case: FinanceBenchCase, chunks: list[DocumentChunk]) -> set[str]:
    normalized = normalize_document_name(case.source_name)
    return {chunk.chunk_id for chunk in chunks if normalize_document_name(chunk.source_name) == normalized and chunk.page in case.evidence_pages}
