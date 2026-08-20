import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings
from app.retrieval.corpus import load_processed_chunks
from app.retrieval.sparse import BM25Retriever
from evaluation.financebench import load_cases, map_case_to_relevant_chunks
from evaluation.metrics.retrieval import recall_at_k, reciprocal_rank

chunks = load_processed_chunks(Path("data/processed/documents"))
retriever = BM25Retriever(chunks)
cases = load_cases(get_settings().financebench_path / "data" / "financebench_open_source.jsonl")
evaluations = []
skipped = 0
for case in cases:
    relevant = map_case_to_relevant_chunks(case, chunks)
    if not relevant:
        skipped += 1
        continue
    hits = retriever.search(case.question, 10)
    evaluations.append((case, hits, relevant))
    if len(evaluations) == 10:
        break
if not evaluations:
    raise SystemExit("No evaluable cases in processed corpus")
print(f"Evaluated queries: {len(evaluations)}\nSkipped queries: {skipped}")
print(f"Recall@5:  {sum(recall_at_k(h, r, 5) for _, h, r in evaluations) / len(evaluations):.4f}")
print(f"Recall@10: {sum(recall_at_k(h, r, 10) for _, h, r in evaluations) / len(evaluations):.4f}")
print(f"MRR:       {sum(reciprocal_rank(h, r) for _, h, r in evaluations) / len(evaluations):.4f}")
for case, hits, relevant in evaluations[:3]:
    first = next((hit.rank for hit in hits if hit.chunk_id in relevant), None)
    print(f"{case.question_id}: first_relevant={first or '>10'} top={hits[0].source_name}:{hits[0].page}")
