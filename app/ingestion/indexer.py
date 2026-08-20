import json
from pathlib import Path

from app.ingestion.chunker import DEFAULT_CHUNKING_CONFIG, ChunkingConfig, chunk_pages
from app.ingestion.cleaner import clean_text
from app.ingestion.financebench import build_document_id, load_document_metadata
from app.ingestion.loaders import load_pdf
from app.ingestion.metadata import DocumentChunk, ParsedPage, file_hash


def process_document(
    path: Path,
    dataset_root: Path,
    output_dir: Path = Path("data/processed/documents"),
    manifest_path: Path = Path("data/dev/manifest.json"),
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> list[DocumentChunk]:
    document_id = build_document_id(path, dataset_root)
    source_hash = file_hash(str(path))
    pages = [
        ParsedPage(page_number=page.page_number, text=clean_text(page.text))
        for page in load_pdf(path)
    ]
    chunks = chunk_pages(
        pages, document_id, path.name, load_document_metadata(dataset_root, path), config
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{document_id}.jsonl"
    temporary = output_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in chunks:
            output.write(chunk.model_dump_json() + "\n")
    temporary.replace(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    documents = manifest.setdefault("documents", {})
    documents[document_id] = {
        "source_name": path.name,
        "source_path": path.resolve().relative_to(dataset_root.resolve()).as_posix(),
        "document_hash": source_hash,
        "chunk_count": len(chunks),
    }
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_temporary.replace(manifest_path)
    return chunks
