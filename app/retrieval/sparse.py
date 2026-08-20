import math
import re
from collections import Counter
from collections.abc import Sequence

from app.ingestion.metadata import DocumentChunk
from app.retrieval.types import RetrievalHit

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*%?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: Sequence[DocumentChunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.document_frequency = Counter(token for tokens in self.tokens for token in set(tokens))

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        if top_k <= 0 or not self.chunks or not tokenize(query):
            return []
        query_terms = Counter(tokenize(query))
        scores: list[tuple[float, int]] = []
        for index, tokens in enumerate(self.tokens):
            frequency = Counter(tokens)
            score = 0.0
            for term, query_frequency in query_terms.items():
                df = self.document_frequency.get(term, 0)
                if not df:
                    continue
                idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
                tf = frequency[term]
                denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                score += query_frequency * idf * (tf * (self.k1 + 1) / denominator)
            scores.append((score, index))
        ranked = sorted(scores, key=lambda item: (-item[0], self.chunks[item[1]].chunk_id))[:top_k]
        return [RetrievalHit(chunk_id=self.chunks[index].chunk_id, document_id=self.chunks[index].document_id, text=self.chunks[index].text, source_name=self.chunks[index].source_name, page=self.chunks[index].page, score=score, rank=rank, metadata=self.chunks[index].metadata) for rank, (score, index) in enumerate(ranked, start=1)]
