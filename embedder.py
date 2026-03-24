from sentence_transformers import SentenceTransformer
import functools

@functools.lru_cache(maxsize=1)
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-large")

def get_embeddings(chunks):
    model = load_model()
    prefixed_chunks = []
    for chunk in chunks:
        prefixed_chunks.append("passage: " + chunk)
    return model.encode(prefixed_chunks)

def get_query_embedding(query):
    model = load_model()
    return model.encode("query: " + query)