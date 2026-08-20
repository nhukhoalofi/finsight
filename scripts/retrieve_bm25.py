import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval.corpus import load_processed_chunks
from app.retrieval.sparse import BM25Retriever

parser = argparse.ArgumentParser()
parser.add_argument("--query", required=True)
parser.add_argument("--top-k", type=int, default=10)
args = parser.parse_args()
for hit in BM25Retriever(load_processed_chunks(Path("data/processed/documents"))).search(args.query, args.top_k):
    print(f"#{hit.rank} {hit.score:.3f} {hit.chunk_id} {hit.source_name} p.{hit.page}: {hit.text[:160]}")
