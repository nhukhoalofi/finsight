import json
from pathlib import Path

import pytest

from evaluation.datasets.retrieval import load_retrieval_eval_cases


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def valid_record(identifier: str = "case-1") -> dict[str, object]:
    return {
        "id": identifier,
        "question": "What was revenue?",
        "source_name": "example",
        "evidence_pages": [1],
        "relevant_chunk_ids": ["chunk-1"],
        "answerable": True,
    }


def test_loader_preserves_order_and_reads_utf8(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    first, second = valid_record("first"), valid_record("second")
    second["question"] = "Doanh thu là gì?"
    write_jsonl(path, [first, second])
    assert [case.id for case in load_retrieval_eval_cases(path)] == ["first", "second"]


def test_loader_rejects_duplicate_ids_with_line_context(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    write_jsonl(path, [valid_record(), valid_record()])
    with pytest.raises(ValueError, match=r"Duplicate.*:2"):
        load_retrieval_eval_cases(path)


@pytest.mark.parametrize(
    "contents, message",
    [
        ("{bad json}\n", r"Invalid.*:1"),
        (json.dumps({**valid_record(), "relevant_chunk_ids": []}) + "\n", r"Invalid.*:1"),
    ],
)
def test_loader_rejects_malformed_or_positive_cases_without_gold(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_retrieval_eval_cases(path)

