from collections import Counter
from src.retrieval.vector_store import get_client

collection = get_client().get_collection("compliance_regs")
data = collection.get(include=["metadatas"])

print("Total chunks stored:", len(data["metadatas"]))
print("Chunks per source:", Counter(m["source"] for m in data["metadatas"]))