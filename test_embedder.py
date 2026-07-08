import time
from src.ingestion.pdf_parser import extract_text
from src.ingestion.chunker import chunk_text
from src.retrieval.embedder import get_embeddings

#Fixed chunks testing
print("Extracting and chunking GDPR...")
text = extract_text("data/sample_contracts/gdpr.pdf")
chunks = chunk_text(text, "GDPR")

# Test with 10 chunks
test_chunks = [c["text"] for c in chunks[:10]]

# Embedding 10 chunks
print("Embedding 10 chunks...")
start = time.time()
embeddings = get_embeddings(test_chunks)
end = time.time()

#Show Embeddings process
print(f"Time taken: {end - start:.2f} seconds")
print(f"Embedding shape: {embeddings.shape}")
print(f"First embedding (first 5 numbers): {embeddings[0][:5]}")