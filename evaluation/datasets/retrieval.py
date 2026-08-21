"""Typed, frozen retrieval-evaluation dataset contracts."""

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError, model_validator


class RetrievalEvalCase(BaseModel):
    id: str
    question: str
    source_name: str
    evidence_pages: list[int]
    relevant_chunk_ids: list[str]
    answerable: bool = True
    dataset_source: str = "financebench_open_source"
    page_mapping_version: str = "financebench_page_plus_one"

    @model_validator(mode="after")
    def require_gold_for_answerable_case(self) -> "RetrievalEvalCase":
        if self.answerable and not self.relevant_chunk_ids:
            raise ValueError("answerable retrieval cases require relevant_chunk_ids")
        return self


def load_retrieval_eval_cases(path: Path) -> list[RetrievalEvalCase]:
    """Load ordered UTF-8 JSONL and reject malformed or duplicate records."""
    cases: list[RetrievalEvalCase] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"Empty JSONL record at {path}:{line_number}")
            try:
                case = RetrievalEvalCase.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as error:
                raise ValueError(f"Invalid retrieval evaluation record at {path}:{line_number}") from error
            if case.id in seen_ids:
                raise ValueError(f"Duplicate retrieval evaluation id {case.id!r} at {path}:{line_number}")
            seen_ids.add(case.id)
            cases.append(case)
    return cases
