from src.retrieval.embedder import get_query_embedding
from src.retrieval.vector_store import hybrid_search
from src.retrieval.reranker import rerank

def retrieve(query, top_k=3):
    """
    Full retrieval pipeline:
    1. Embed query
    2. Hybrid search (dense + BM25)
    3. Rerank with CrossEncoder
    4. Return top k chunks
    """
    # Step 1 - embed query
    query_embedding = get_query_embedding(query)
    
    # Step 2 - hybrid search
    candidate_docs, candidate_metas = hybrid_search(
        query, query_embedding, k=10
    )
    
    # Step 3 - rerank
    top_docs, top_metas, top_scores = rerank(
        query, candidate_docs, candidate_metas, top_k=top_k
    )
    
    return top_docs, top_metas, top_scores