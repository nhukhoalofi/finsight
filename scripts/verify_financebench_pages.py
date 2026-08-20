import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings


def normalize(text: str) -> str:
    """Normalize presentation differences without altering evidence semantics."""
    text = unicodedata.normalize("NFKC", text).replace("-\n", "")
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def anchors(text: str) -> list[str]:
    words = normalize(text).split()
    if len(words) < 8:
        return []
    starts = (0, max(0, len(words) // 2 - 5), max(0, len(words) - 10))
    return list(dict.fromkeys(" ".join(words[start : start + 10]) for start in starts))


def match_evidence(evidence: dict[str, Any], pages: list[str]) -> tuple[int | None, str, float]:
    full_page = normalize(str(evidence.get("evidence_text_full_page") or ""))
    excerpt = normalize(str(evidence.get("evidence_text") or ""))
    for candidate, method in ((full_page, "full_page_exact"), (excerpt, "excerpt_exact")):
        if len(candidate) >= 80:
            match = next((index for index, page in enumerate(pages, start=1) if candidate in page), None)
            if match is not None:
                return match, method, 1.0

    evidence_anchors = anchors(str(evidence.get("evidence_text") or ""))
    for index, page in enumerate(pages, start=1):
        matches = sum(anchor in page for anchor in evidence_anchors)
        if matches >= 2:
            return index, "two_anchor_exact", matches / len(evidence_anchors)
    return None, "unresolved", 0.0


def write_artifact(records: list[dict[str, Any]]) -> Path:
    artifact = Path("artifacts/m3/page_alignment_diagnostic.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-documents", type=int, default=3)
    parser.add_argument("--max-evidence", type=int, default=30)
    args = parser.parse_args()

    root = get_settings().financebench_path
    source = root / "data" / "financebench_open_source.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if (root / "pdfs" / f"{record['doc_name']}.pdf").exists() and record.get("evidence"):
            grouped[record["doc_name"]].append(record)
    results: list[dict[str, Any]] = []
    for document_index, (doc_name, cases) in enumerate(grouped.items(), start=1):
        if document_index > args.max_documents or len(results) >= args.max_evidence:
            break
        pages = [
            normalize(page.extract_text() or "")
            for page in PdfReader(root / "pdfs" / f"{doc_name}.pdf").pages
        ]
        for case in cases:
            for evidence in case["evidence"]:
                if len(results) >= args.max_evidence:
                    break
                matched, method, score = match_evidence(evidence, pages)
                results.append(
                    {
                        "financebench_id": case["financebench_id"],
                        "doc_name": doc_name,
                        "question": case["question"],
                        "evidence_page_num": evidence["evidence_page_num"],
                        "matched_pypdf_page": matched,
                        "offset": matched - evidence["evidence_page_num"] if matched is not None else None,
                        "match_method": method,
                        "score": score,
                    }
                )
        write_artifact(results)

    artifact = write_artifact(results)
    verified = sum(record["matched_pypdf_page"] is not None for record in results)
    print(f"Wrote {len(results)} records ({verified} verified) to {artifact}")


if __name__ == "__main__":
    main()
