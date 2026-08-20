from app.retrieval.types import RetrievalHit


def recall_at_k(hits: list[RetrievalHit], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len({hit.chunk_id for hit in hits[:k]} & relevant_ids) / len(relevant_ids)


def reciprocal_rank(hits: list[RetrievalHit], relevant_ids: set[str]) -> float:
    for hit in hits:
        if hit.chunk_id in relevant_ids:
            return 1 / hit.rank
    return 0.0
