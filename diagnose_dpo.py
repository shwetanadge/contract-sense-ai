from src.retrieval.embedder import get_query_embedding
from src.retrieval.vector_store import get_client, hybrid_search
from src.retrieval.reranker import get_reranker

PHRASE = "regular and systematic monitoring"   # key text from Art. 37(1)(b)

# 1. Which stored chunks contain the key phrase?
data = get_client().get_collection("compliance_regs").get(
    include=["documents", "metadatas"]
)
targets = [
    (m["chunk_id"], d)
    for d, m in zip(data["documents"], data["metadatas"])
    if m["source"] == "GDPR" and PHRASE in d.lower().replace("\n", " ")
]
print(f"Chunks containing '{PHRASE}': {len(targets)}")
for cid, d in targets:
    print(f"  chunk {cid}: {d[:150]!r}")

# 2. Where do they land in search and reranking?
model = get_reranker()
for q in ["When is a DPO required?", "When must a Data Protection Officer be appointed?"]:
    docs, metas = hybrid_search(q, get_query_embedding(q), k=30)
    scores = model.predict([[q, d] for d in docs])
    ranked = sorted(zip(scores, metas, docs), key=lambda x: x[0], reverse=True)

    print(f"\nQUESTION: {q}")
    print(f"Candidates found: {len(docs)}")
    for rank, (s, m, d) in enumerate(ranked[:6], start=1):
        flag = "  <-- ART. 37(1) TEXT" if any(d == t[1] for t in targets) else ""
        print(f"  #{rank} {s:6.2f} chunk {m['chunk_id']}{flag}")
    in_candidates = any(d == t[1] for t in targets for d in docs)
    print("Art. 37(1) chunk among candidates:", in_candidates)