from src.ingestion.pdf_parser import extract_text
from src.ingestion.chunker import chunk_text

# Extract text from GDPR
print("Extracting GDPR text...")
gdpr_text = extract_text("data/sample_contracts/gdpr.pdf")

# Chunk the text
print("Chunking text...")
chunks = chunk_text(gdpr_text, "GDPR")

# Print total chunks
print(f"\nTotal chunks: {len(chunks)}")

# Print chunks 5, 10, and 15
for i in [5, 10, 15]:
    print(f"\n=== CHUNK {i} ===")
    print(chunks[i]["text"])
    print(f"Metadata: {chunks[i]['metadata']}")