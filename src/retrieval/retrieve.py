from src.retrieval.embedder import get_query_embedding
from src.retrieval.vector_store import hybrid_search, get_chunk
from src.retrieval.reranker import rerank

def retrieve(query, top_k=3, expand_top=5):
    """
    Retrieval pipeline:
    1. Embed query
    2. Hybrid search (dense + BM25)
    3. First rerank, keep the best few chunks
    4. Add the chunk before and after each of them (neighbour expansion)
    5. Rerank again and return the top k
    """
    query_embedding = get_query_embedding(query)

    candidate_docs, candidate_metas = hybrid_search(
        query, query_embedding, k=30
    )

    # First pass: find the best few chunks
    _, first_metas, _ = rerank(
        query, candidate_docs, candidate_metas, top_k=expand_top
    )

    # Add neighbours of those chunks to the candidate pool
    seen = set(candidate_docs)
    for meta in first_metas:
        for step in (-1, 1):
            neighbour = get_chunk(meta["source"], meta["chunk_id"] + step)
            if neighbour and neighbour["text"] not in seen:
                seen.add(neighbour["text"])
                candidate_docs.append(neighbour["text"])
                candidate_metas.append(neighbour["metadata"])

    # Second pass: final ranking over the enlarged pool
    return rerank(query, candidate_docs, candidate_metas, top_k=top_k)