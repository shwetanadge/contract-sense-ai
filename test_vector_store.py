import time
from src.ingestion.pdf_parser import extract_text
from src.ingestion.chunker import chunk_text
from src.retrieval.embedder import get_embeddings, get_query_embedding
from src.retrieval.vector_store import build_index, search
from src.storage.db import init_db, insert_chunks, get_query_logs, log_query

# Initialize database
print("Initializing database...")
init_db()

# Process GDPR
print("Extracting GDPR text...")
text = extract_text("data/sample_contracts/gdpr.pdf")
chunks = chunk_text(text, "GDPR")
print(f"Total chunks: {len(chunks)}")

# Embed all chunks
print("Embedding all chunks... (this takes 10-15 mins on CPU)")
texts = [c['text'] for c in chunks]
embeddings = get_embeddings(texts)

# Store in ChromaDB
print("Building ChromaDB index...")
build_index(chunks, embeddings)

# Store in SQLite
print("Storing chunks in SQLite...")
insert_chunks(chunks)

# Test search
print("\nTesting search...")
query = "When is a Data Protection Officer required?"
query_emb = get_query_embedding(query)
results = search(query_emb, k=3)

print(f"\nQuery: {query}")
print("\nTop 3 results:")
for i, doc in enumerate(results['documents'][0]):
    print(f"\n--- Result {i+1} ---")
    print(doc[:300])
    print(f"Source: {results['metadatas'][0][i]}")

# Test query logging
start = time.time()
fake_answer = "A Data Protection Officer is required under Article 37..."
latency = time.time() - start
log_query(query, fake_answer, latency)
log_query("What is GDPR?", "GDPR is a regulation...", 0.9)
log_query("What is Article 17?", "Article 17 is...", 1.1)

# Verify logs
print("\n=== QUERY LOG ===")
logs = get_query_logs()
for log in logs:
    print(log)