from src.ingestion.pdf_parser import extract_text
from src.ingestion.chunker import chunk_text
from src.retrieval.embedder import get_embeddings
from src.retrieval.vector_store import build_index
from src.storage.db import init_db, insert_chunks
import sqlite3

DOCS = [
    ("data/sample_contracts/gdpr.pdf", "GDPR"),
    ("data/sample_contracts/HGB.pdf", "HGB"),
]

all_chunks = []
for path, source in DOCS:
    print(f"Reading {source}...")
    text = extract_text(path)
    chunks = chunk_text(text, source)
    print(f"  {source}: {len(chunks)} chunks")
    all_chunks.extend(chunks)

print("\nSample HGB chunk:")
hgb_sample = [c for c in all_chunks if c["metadata"]["source"] == "HGB"][10]
print(hgb_sample["text"][:400])

print(f"\nEmbedding {len(all_chunks)} chunks (this takes a while on CPU)...")
embeddings = get_embeddings([c["text"] for c in all_chunks])

build_index(all_chunks, embeddings)

# Refresh the SQLite chunks table so it has no duplicates
init_db()
conn = sqlite3.connect("compliance_rag.db")
conn.execute("DELETE FROM chunks")
conn.commit()
conn.close()
insert_chunks(all_chunks)

print("\nDone!")