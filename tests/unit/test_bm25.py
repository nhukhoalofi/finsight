from app.ingestion.metadata import DocumentChunk
from app.retrieval.sparse import BM25Retriever, tokenize


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(chunk_id=chunk_id, document_id="doc", text=text, source_name="source.pdf", page=1, ordinal=0, content_hash=chunk_id)


def test_tokenizer_preserves_financial_signals() -> None:
    tokens = tokenize("Revenue increased 12.5% to $4.2 billion in 2024 Form 10-K for 3M")
    assert {"revenue", "12.5%", "4.2", "2024", "10-k", "3m"} <= set(tokens)


def test_bm25_ranks_exact_financial_match_and_preserves_provenance() -> None:
    target = make_chunk("a", "Net sales were $32.8 billion in 2015.")
    other = make_chunk("b", "Operating expenses increased during the year.")
    hits = BM25Retriever([target, other]).search("net sales 2015", 10)
    assert [hit.chunk_id for hit in hits][:1] == ["a"]
    assert hits[0].rank == 1 and hits[0].text == target.text and hits[0].page == 1


def test_bm25_is_deterministic_and_handles_edge_cases() -> None:
    retriever = BM25Retriever([make_chunk("b", "rare alpha"), make_chunk("a", "rare alpha")])
    first = retriever.search("rare", 10)
    assert [hit.chunk_id for hit in first] == ["a", "b"]
    assert first == retriever.search("rare", 10)
    assert retriever.search("", 10) == []
    assert retriever.search("rare", 0) == []
    assert BM25Retriever([]).search("rare") == []
