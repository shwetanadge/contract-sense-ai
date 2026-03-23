from sentence_transformers import SentenceTransformer

def get_embeddings(chunks):
# 1. Load a pretrained Sentence Transformer model
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    prefixed_chunks = []
    for chunk in chunks:
        prefixed_chunks.append("passage:" + chunk)