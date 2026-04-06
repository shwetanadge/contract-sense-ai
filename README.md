# Contract Sense AI — Compliance RAG Assistant

An AI-powered compliance assistant that answers regulatory questions 
using Retrieval-Augmented Generation (RAG).

## What it does
- Parses legal PDF documents (GDPR, HGB)
- Chunks and embeds text using multilingual-e5-large
- Stores vectors in ChromaDB for semantic search
- Uses hybrid search (BM25 + dense) with CrossEncoder reranking
- Generates cited answers using LangGraph + Groq (Llama 3)
- Logs all queries to SQLite for analysis

## Tech Stack
- **LLM**: Llama 3 via Groq API
- **Embeddings**: intfloat/multilingual-e5-large
- **Vector Store**: ChromaDB
- **Search**: Hybrid BM25 + Dense + CrossEncoder Reranker
- **Agent**: LangGraph 2-node pipeline
- **Storage**: SQLite
- **PDF Parsing**: pdfplumber

## Project Structure
```
src/
├── ingestion/      # PDF parsing and chunking
├── retrieval/      # Embeddings, vector store, reranker
├── analysis/       # LangGraph RAG agent
└── storage/        # SQLite database
```

## Setup
```bash
git clone https://github.com/shwetanadge/contract-sense-ai
cd contract-sense-ai
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

## Usage
```python
from src.analysis.rag_agent import ask
answer, citations = ask("When is a DPO required?")
print(answer)
```

## Sample Output
Q: When is a DPO required?
A: According to GDPR Article 37, a DPO is required when...
Citations: [1] GDPR chunk 531