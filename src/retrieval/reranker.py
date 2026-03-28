from sentence_transformers import CrossEncoder

# Load reranker model
reranker = None

def get_reranker():
    global reranker
    if reranker is None:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return reranker

def rerank(query, candidate_docs, candidate_metas, top_k=3):
    model = get_reranker()
    
    # Create query-document pairs
    pairs = [[query, doc] for doc in candidate_docs]
    
    # Score each pair
    scores = model.predict(pairs)
    
    # Sort by score
    ranked = sorted(
        zip(scores, candidate_docs, candidate_metas),
        reverse=True
    )
    
    # Return top k
    top_docs = [doc for _, doc, _ in ranked[:top_k]]
    top_metas = [meta for _, _, meta in ranked[:top_k]]
    top_scores = [float(score) for score, _, _ in ranked[:top_k]]
    
    return top_docs, top_metas, top_scores