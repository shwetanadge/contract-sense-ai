from src.retrieval.embedder import get_query_embedding
from src.retrieval.vector_store import hybrid_search, get_chunk
from src.retrieval.reranker import rerank
import re

# Abbreviations users type that the regulation texts spell out in full
ABBREVIATIONS = {
    "DPO": "data protection officer",
    "DPIA": "data protection impact assessment",
    "DSAR": "data subject access request",
    "HGB": "Handelsgesetzbuch",
}

def expand_abbreviations(query):
    for short, full in ABBREVIATIONS.items():
        pattern = re.compile(rf"\b{short}\b")
        # only expand if the full form isn't already in the question
        if pattern.search(query) and full.lower() not in query.lower():
            query = pattern.sub(f"{short} ({full})", query, count=1)
    return query

def add_same_article_neighbours(docs, metas, scores, anchors=2, max_total=8):
    """Add the chunk before and after the best chunks if they belong to the same article."""
    docs, metas, scores = list(docs), list(metas), list(scores)
    seen = set(docs)
    for meta in metas[:anchors]:
        if not meta.get("section"):
            continue
        for step in (-1, 1):
            if len(docs) >= max_total:
                break
            nb = get_chunk(meta["source"], meta["chunk_id"] + step)
            if (nb and nb["metadata"].get("section") == meta["section"]
                    and nb["text"] not in seen):
                seen.add(nb["text"])
                docs.append(nb["text"])
                metas.append(nb["metadata"])
                scores.append(float("nan"))   # not ranked, added for context
    return docs, metas, scores

def retrieve(query, top_k=3, expand_top=5, add_siblings=False):
    """
    Retrieval pipeline:
    1. Embed query
    2. Hybrid search (dense + BM25)
    3. First rerank, keep the best few chunks
    4. Add the chunk before and after each of them (neighbour expansion)
    5. Rerank again and return the top k
    """
    query = expand_abbreviations(query)
    query_embedding = get_query_embedding(query)
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
    top_docs, top_metas, top_scores = rerank(
        query, candidate_docs, candidate_metas, top_k=top_k
    )
    if add_siblings:
        top_docs, top_metas, top_scores = add_same_article_neighbours(
            top_docs, top_metas, top_scores
        )
    return top_docs, top_metas, top_scores