from collections import Counter
from src.retrieval.embedder import get_query_embedding
from src.retrieval.vector_store import hybrid_search
from src.retrieval.reranker import get_reranker

questions = [
    "What is the retention period for accounting records under HGB?",
    "Welche Aufbewahrungsfristen gelten für Handelsbücher?",
]

model = get_reranker()

for q in questions:
    docs, metas = hybrid_search(q, get_query_embedding(q), k=10)
    scores = model.predict([[q, d] for d in docs])

    print("\nQUESTION:", q)
    print("Candidates found, by source:", dict(Counter(m["source"] for m in metas)))
    print("Top 5 after reranking:")
    ranked = sorted(zip(scores, metas, docs), key=lambda x: x[0], reverse=True)
    for s, m, d in ranked[:5]:
        print(f"  {s:6.2f}  [{m['source']}]  {d[:70]!r}")