import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.ingestion.financebench import discover_financebench_pdfs
from app.ingestion.indexer import process_document
from app.ingestion.metadata import file_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic FinanceBench JSONL chunks.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()
    root = get_settings().financebench_path
    paths = [args.path] if args.path else discover_financebench_pdfs(root)
    if args.limit is not None:
        paths = paths[: args.limit]
    total_chunks = 0
    for path in paths:
        chunks = process_document(path, root)
        total_chunks += len(chunks)
        print(f"{path.name}: pages/chunks={len(chunks)} document_hash={file_hash(str(path))} output={chunks[0].document_id if chunks else 'empty'}")
    print(f"Processed {len(paths)} document(s), created {total_chunks} chunk(s).")


if __name__ == "__main__":
    main()
