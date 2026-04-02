from src.ingestion.pdf_parser import extract_text
from src.ingestion.chunker import chunk_text
from src.retrieval.embedder import get_query_embedding
from src.retrieval.vector_store import load_bm25, dense_search
from src.retrieval.retrieve import retrieve

# Load chunks for BM25.
print("Loading chunks for BM25...")
text = extract_text("data/sample_contracts/gdpr.pdf")
chunks = chunk_text(text, "GDPR")
load_bm25(chunks)

# Test query
query = "DPO requirements for 20 employees"
print(f"\nQuery: {query}")

# Dense only results
print("\n=== DENSE ONLY (top 3) ===")
query_emb = get_query_embedding(query)
dense_docs, dense_metas = dense_search(query_emb, k=3)
for i, doc in enumerate(dense_docs):
    print(f"\nResult {i+1}: {doc[:200]}")

# Hybrid + reranked results
print("\n=== HYBRID + RERANKED (top 3) ===")
top_docs, top_metas, top_scores = retrieve(query)
for i, (doc, score) in enumerate(zip(top_docs, top_scores)):
    print(f"\nResult {i+1} (score: {score:.3f}):")
    print(doc[:200])
    print(f"Source: {top_metas[i]}")