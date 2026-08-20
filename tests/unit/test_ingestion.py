import json
from pathlib import Path

import pytest

from app.ingestion import indexer
from app.ingestion.chunker import ChunkingConfig, chunk_pages
from app.ingestion.cleaner import clean_text
from app.ingestion.loaders import load_pdf
from app.ingestion.metadata import ParsedPage, content_hash


def _write_two_page_pdf(path: Path) -> None:
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents 4 0 R >>",
        "<< /Length 38 >>\nstream\nBT /F1 12 Tf 72 720 Td (First page) Tj ET\nendstream",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R >>",
        "<< /Length 39 >>\nstream\nBT /F1 12 Tf 72 720 Td (Second page) Tj ET\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, object_value in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n{object_value}\nendobj\n".encode())
    startxref = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    content.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    content.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode())
    path.write_bytes(content)


def _fake_loader(path: Path) -> list[ParsedPage]:
    return [ParsedPage(page_number=1, text=path.read_text(encoding="utf-8"))]


def test_cleaner_preserves_financial_values() -> None:
    assert clean_text("Revenue  $4.2 billion\n\n rose 12.5% in 2024.") == "Revenue $4.2 billion\n\nrose 12.5% in 2024."


def test_chunking_is_stable_nonempty_and_bounded() -> None:
    pages = [ParsedPage(page_number=1, text=" ".join(f"word{i}" for i in range(20)))]
    config = ChunkingConfig(target_size=8, overlap=2, hard_limit=10)
    first = chunk_pages(pages, "document", "sample.pdf", {}, config)
    second = chunk_pages(pages, "document", "sample.pdf", {}, config)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.ordinal for chunk in first] == list(range(len(first)))
    assert all(chunk.text.strip() and len(chunk.text.split()) <= 8 for chunk in first)
    assert first[0].text.split() == [f"word{i}" for i in range(8)]
    assert second[1].text.split()[:2] == ["word6", "word7"]


@pytest.mark.parametrize(
    "config, message",
    [
        (ChunkingConfig(target_size=0), "target_size must be positive"),
        (ChunkingConfig(target_size=8, overlap=-1), "overlap must be non-negative"),
        (ChunkingConfig(target_size=8, overlap=8), "overlap must be smaller than target_size"),
        (ChunkingConfig(target_size=8, overlap=9), "overlap must be smaller than target_size"),
        (
            ChunkingConfig(target_size=8, overlap=2, hard_limit=7),
            "hard_limit must be >= target_size",
        ),
    ],
)
def test_chunking_rejects_invalid_config(config: ChunkingConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_pages([ParsedPage(page_number=1, text="one two")], "document", "sample.pdf", {}, config)


def test_content_hash_changes_with_content() -> None:
    assert content_hash("same text") == content_hash("same   text")
    assert content_hash("same text") != content_hash("changed text")


def test_loader_preserves_page_order_and_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "two-pages.pdf"
    _write_two_page_pdf(pdf_path)

    pages = load_pdf(pdf_path)

    assert [page.page_number for page in pages] == [1, 2]
    assert len(pages) == 2
    assert "First page" in pages[0].text
    assert "Second page" in pages[1].text


def test_processed_output_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.pdf"
    source.write_text("stable financial content", encoding="utf-8")
    monkeypatch.setattr(indexer, "load_pdf", _fake_loader)
    output_dir, manifest = tmp_path / "output", tmp_path / "manifest.json"

    first = indexer.process_document(source, tmp_path, output_dir, manifest)
    second = indexer.process_document(source, tmp_path, output_dir, manifest)

    output = next(output_dir.glob("*.jsonl"))
    stored = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    entries = json.loads(manifest.read_text(encoding="utf-8"))["documents"]
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len(stored) == len(first)
    assert list(entries) == [first[0].document_id]
    assert entries[first[0].document_id]["chunk_count"] == len(first)


def test_changed_source_replaces_processed_output(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.pdf"
    monkeypatch.setattr(indexer, "load_pdf", _fake_loader)
    output_dir, manifest = tmp_path / "output", tmp_path / "manifest.json"
    source.write_text("old content", encoding="utf-8")
    first = indexer.process_document(source, tmp_path, output_dir, manifest)
    first_hash = json.loads(manifest.read_text(encoding="utf-8"))["documents"][
        first[0].document_id
    ]["document_hash"]
    source.write_text("new content with a changed chunk", encoding="utf-8")
    second = indexer.process_document(source, tmp_path, output_dir, manifest)

    output = next(output_dir.glob("*.jsonl"))
    stored_ids = [json.loads(line)["chunk_id"] for line in output.read_text(encoding="utf-8").splitlines()]
    entry = json.loads(manifest.read_text(encoding="utf-8"))["documents"][first[0].document_id]
    assert {chunk.chunk_id for chunk in first}.isdisjoint(chunk.chunk_id for chunk in second)
    assert stored_ids == [chunk.chunk_id for chunk in second]
    assert entry["document_hash"] != first_hash
    assert entry["chunk_count"] == len(second)
