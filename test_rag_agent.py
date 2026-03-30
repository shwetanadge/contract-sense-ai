from src.ingestion.pdf_parser import extract_text
from src.ingestion.chunker import chunk_text
from src.retrieval.vector_store import load_bm25
from src.analysis.rag_agent import ask
from src.storage.db import get_query_logs

# Load BM25 index
print("Loading BM25 index...")
text = extract_text("data/sample_contracts/gdpr.pdf")
chunks = chunk_text(text, "GDPR")
load_bm25(chunks)

# Test questions
questions = [
    "When is a DPO required?",
    "What is the capital of France?",
    "GDPR data breach notification deadline?",
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    answer, citations = ask(question)
    print(f"A: {answer}")
    print(f"Citations: {citations}")

# Verify SQL logs
print(f"\n{'='*60}")
print("=== QUERY LOG ===")
logs = get_query_logs()
for log in logs[-3:]:
    print(f"Q: {log[1][:50]} | Latency: {log[4]}s")