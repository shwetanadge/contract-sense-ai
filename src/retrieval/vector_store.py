import chromadb
from chromadb.config import Settings

def get_client():
    return chromadb.PersistentClient(path="./chroma_db")

def build_index(chunks, embeddings):
    client = get_client()
    
    # Delete existing collection if exists
    try:
        client.delete_collection("compliance_regs")
    except:
        pass
    
    collection = client.create_collection("compliance_regs")
    
    # Prepare data
    ids = [f"{c['metadata']['source']}_{c['metadata']['chunk_id']}" for c in chunks]
    documents = [c['text'] for c in chunks]
    metadatas = [c['metadata'] for c in chunks]
    embeddings_list = embeddings.tolist()
    
    # Add in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            embeddings=embeddings_list[i:i+batch_size]
        )
    
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return collection

def search(query_embedding, k=3):
    client = get_client()
    collection = client.get_collection("compliance_regs")
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )
    
    return results