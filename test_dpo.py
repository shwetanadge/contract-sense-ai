from src.retrieval.retrieve import retrieve

for q in ["When is a DPO required?", "When must a Data Protection Officer be appointed?"]:
    docs, metas, scores = retrieve(q, top_k=5)
    print("\nQUESTION:", q)
    for m, s in zip(metas, scores):
        print(f"  {s:6.2f}  {m['source']} chunk {m['chunk_id']}")