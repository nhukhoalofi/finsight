import hashlib
import json
from pathlib import Path
from typing import Any


def discover_financebench_pdfs(root: Path) -> list[Path]:
    pdf_root = root / "pdfs" if (root / "pdfs").is_dir() else root
    return sorted(path for path in pdf_root.rglob("*.pdf") if path.is_file())


def build_document_id(path: Path, root: Path) -> str:
    relative_path = path.resolve().relative_to(root.resolve()).as_posix().lower()
    return "financebench-" + hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]


def load_document_metadata(root: Path, path: Path) -> dict[str, Any]:
    metadata_path = root / "data" / "financebench_document_information.jsonl"
    if not metadata_path.is_file():
        return {}
    document_name = path.stem
    with metadata_path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("doc_name") == document_name:
                return {
                    key: record[key]
                    for key in ("company", "gics_sector", "doc_type", "doc_period")
                    if key in record
                }
    return {}
