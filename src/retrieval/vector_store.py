import chromadb
from rank_bm25 import BM25Okapi
from src.retrieval.embedder import get_embeddings

# Global BM25 index
bm25_index = None
bm25_chunks = None

def get_client():
    return chromadb.PersistentClient(path="./chroma_db")

def build_index(chunks, embeddings):
    global bm25_index, bm25_chunks
    
    client = get_client()
    
    try:
        client.delete_collection("compliance_regs")
    except:
        pass
    
    collection = client.create_collection("compliance_regs")
    
    ids = [f"{c['metadata']['source']}_{c['metadata']['chunk_id']}" for c in chunks]
    documents = [c['text'] for c in chunks]
    metadatas = [c['metadata'] for c in chunks]
    embeddings_list = embeddings.tolist()
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            embeddings=embeddings_list[i:i+batch_size]
        )
    
    # Build BM25 index
    bm25_chunks = chunks
    tokenized = [c['text'].lower().split() for c in chunks]
    bm25_index = BM25Okapi(tokenized)
    
    print(f"Stored {len(chunks)} chunks in ChromaDB and BM25")
    return collection

def load_bm25(chunks):
    global bm25_index, bm25_chunks
    bm25_chunks = chunks
    tokenized = [c['text'].lower().split() for c in chunks]
    bm25_index = BM25Okapi(tokenized)

def dense_search(query_embedding, k=10):
    client = get_client()
    collection = client.get_collection("compliance_regs")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )
    return results['documents'][0], results['metadatas'][0]

def bm25_search(query, k=10):
    global bm25_index, bm25_chunks
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), 
                        key=lambda i: scores[i], 
                        reverse=True)[:k]
    docs = [bm25_chunks[i]['text'] for i in top_indices]
    metas = [bm25_chunks[i]['metadata'] for i in top_indices]
    return docs, metas

def hybrid_search(query, query_embedding, k=10):
    # Get results from both
    dense_docs, dense_metas = dense_search(query_embedding, k)
    bm25_docs, bm25_metas = bm25_search(query, k)
    
    # Merge and deduplicate
    seen = set()
    combined_docs = []
    combined_metas = []
    
    for doc, meta in zip(dense_docs + bm25_docs, 
                         dense_metas + bm25_metas):
        if doc not in seen:
            seen.add(doc)
            combined_docs.append(doc)
            combined_metas.append(meta)
    
    return combined_docs, combined_metas

def search(query_embedding, k=3):
    client = get_client()
    collection = client.get_collection("compliance_regs")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )
    return results